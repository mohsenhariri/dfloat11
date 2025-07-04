import torch
import math
import os
import re
from sys import stderr
from typing import Optional, Dict, Union
from tqdm import tqdm

import torch
import torch.nn as nn

import dfloat11_decode_v2

from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoConfig, GenerationConfig
from transformers.modeling_utils import no_init_weights


class TensorManager:
    """
    Static utility class that manages tensor allocation and reuse
    to minimize memory allocation overhead during tensor reconstruction.
    """
    # Static class variable to store tensors for each device
    _tensors = {}  # Maps device to tensor

    @staticmethod
    def get_tensor(device, n_elements):
        """
        Get a bfloat16 tensor with at least n_elements on the specified device.

        If a tensor already exists on the device and is larger than n_elements,
        a slice of the tensor with exactly n_elements is returned. If n_elements 
        exceeds the size of the existing tensor, the existing tensor is deallocated 
        and a larger one is allocated.

        Args:
            device: The device to allocate the tensor on (e.g., 'cuda:0')
            n_elements: The exact number of elements required

        Returns:
            A bfloat16 tensor with exactly n_elements on the specified device
        """
        # Convert device to torch.device if it's a string
        if isinstance(device, str):
            device = torch.device(device)
        
        # Check if we already have a tensor for this device
        if device in TensorManager._tensors:
            existing_tensor = TensorManager._tensors[device]
            
            # If existing tensor is large enough, return a slice of it
            if existing_tensor.numel() >= n_elements:
                return existing_tensor[:n_elements]
            
            # Otherwise, delete the existing tensor to free up memory
            del TensorManager._tensors[device]
            torch.cuda.empty_cache()  # Ensure memory is freed
        
        # Allocate a new tensor
        new_tensor = torch.empty(n_elements, dtype=torch.bfloat16, device=device)
        print(f'Allocated {n_elements} bf16 on device {device}', file=stderr)
        
        # Store the tensor
        TensorManager._tensors[device] = new_tensor
        
        return new_tensor

    @staticmethod
    def clear_device(device=None):
        """
        Clear tensors for a specific device or all devices if none specified.
        
        Args:
            device: The device to clear tensors from, or None to clear all devices
        """
        if device is None:
            # Clear all devices
            TensorManager._tensors.clear()
        else:
            # Convert device to torch.device if it's a string
            if isinstance(device, str):
                device = torch.device(device)
                
            # Remove specific device
            if device in TensorManager._tensors:
                del TensorManager._tensors[device]
                
        torch.cuda.empty_cache()  # Ensure memory is freed


def get_hook(threads_per_block, bytes_per_thread):
    """
    Creates a PyTorch forward pre-hook that decodes compressed DFloat11 weights on-the-fly.
    
    This hook reconstructs full-precision weights from compressed representations
    using a custom CUDA kernel during the forward pass.
    
    Args:
        threads_per_block: CUDA thread configuration 
        bytes_per_thread: Number of bytes processed per CUDA thread
        
    Returns:
        A forward pre-hook function for PyTorch modules
    """
    threads_per_block = tuple(threads_per_block)

    def decode_hook(module, _):
        # Get dimensions for tensor reconstruction
        n_elements = module.sign_mantissa.numel()
        n_bytes = module.encoded_exponent.numel()
        n_luts = module.luts.shape[0]

        # Get output tensor for reconstructed weights
        device = module.encoded_exponent.device
        reconstructed = TensorManager.get_tensor(device, n_elements)

        # Configure CUDA grid dimensions for the kernel launch
        blocks_per_grid = (int(math.ceil(n_bytes / (threads_per_block[0] * bytes_per_thread))), )

        # Launch the precompiled CUDA kernel to decode the compressed weights
        dfloat11_decode_v2.decode(
            module.luts.data_ptr(),
            module.encoded_exponent.data_ptr(),
            module.sign_mantissa.data_ptr(),
            module.output_positions.data_ptr(),
            module.gaps.data_ptr(),
            reconstructed.data_ptr(),
            n_luts, n_bytes, n_elements,
            blocks_per_grid[0],
            threads_per_block[0],
            module.shared_mem_size
        )

        # Inject reconstructed weights into the appropriate module
        if isinstance(module, nn.Linear):
            module.weight = reconstructed.view(
                module.out_features, module.in_features
            )
        elif isinstance(module, nn.Embedding):
            module.weight = reconstructed.view(
                module.num_embeddings, module.embedding_dim
            )
        else:
            # Handle special case where weights need to be split across multiple submodules
            weights = torch.tensor_split(reconstructed, module.split_positions)
            for sub_module, weight in zip(module.weight_injection_modules, weights):
                sub_module.weight = weight.view(sub_module.out_features, sub_module.in_features)

    return decode_hook


def load_and_replace_tensors(model, directory_path, dfloat11_config):
    """
    Loads DFloat11 compressed weights from safetensors files and configures the model
    to use them with on-the-fly decompression.
    
    Args:
        model: The PyTorch model to load weights into
        directory_path: Path to the directory containing safetensors files
        dfloat11_config: Configuration for DFloat11 compression
        
    Returns:
        The model with configured DFloat11 compression
    """
    threads_per_block = dfloat11_config['threads_per_block']
    bytes_per_thread  = dfloat11_config['bytes_per_thread']
    pattern_dict      = dfloat11_config['pattern_dict']
    
    # Get all .safetensors files in the directory
    safetensors_files = [f for f in os.listdir(directory_path) if f.endswith('.safetensors')]
    for file_name in tqdm(safetensors_files, desc='Loading DFloat11 safetensors'):
        file_path = os.path.join(directory_path, file_name)
        
        # Load the tensors from the file
        loaded_tensors = load_file(file_path)
        
        # Iterate over each tensor in the file
        for tensor_name, tensor_value in loaded_tensors.items():
            # Check if this tensor exists in the model's state dict
            if tensor_name in model.state_dict():
                # Get the parameter or buffer
                if tensor_name in dict(model.named_parameters()):
                    # It's a parameter, we can set it directly
                    param = dict(model.named_parameters())[tensor_name]
                    if param.shape == tensor_value.shape:
                        param.data.copy_(tensor_value)
                    else:
                        print(f"Shape mismatch for {tensor_name}: model {param.shape} vs loaded {tensor_value.shape}", file=stderr)
                else:
                    # It's a buffer, we can also set it directly
                    buffer = dict(model.named_buffers())[tensor_name]
                    if buffer.shape == tensor_value.shape:
                        buffer.copy_(tensor_value)
                    else:
                        print(f"Shape mismatch for {tensor_name}: model {buffer.shape} vs loaded {tensor_value.shape}", file=stderr)
            else:
                # Split the tensor name to get module path
                parts = tensor_name.split('.')
                module = model
                
                # Navigate to the correct module
                for i, part in enumerate(parts[:-1]):
                    if hasattr(module, part):
                        module = getattr(module, part)
                    else:
                        print(f"Cannot find module path for {tensor_name}", file=stderr)
                        break
                else:
                    if parts[-1] == 'split_positions':
                        setattr(module, 'split_positions', tensor_value.tolist())
                    else:
                        # Register the buffer to the found module
                        module.register_buffer(parts[-1], tensor_value)

                    # Set up decompression for encoded weights
                    if parts[-1] == 'encoded_exponent':
                        # Register the decode hook to decompress weights during forward pass
                        module.register_forward_pre_hook(get_hook(threads_per_block, bytes_per_thread))

                        # Configure weight injection based on module type
                        for pattern, attr_names in pattern_dict.items():
                            if re.fullmatch(pattern, '.'.join(parts[:-1])):
                                if isinstance(module, nn.Embedding):
                                    # Remove weight attribute from embedding layer
                                    tmp = module.weight
                                    delattr(module, 'weight')
                                    del tmp
                                elif isinstance(module, nn.Linear):
                                    # Remove weight attribute from linear layer
                                    tmp = module.weight
                                    delattr(module, 'weight')
                                    del tmp
                                else:
                                    # Handle special case for multi-module weight injection
                                    setattr(module, 'weight_injection_modules', [])
                                    for attr_path in attr_names:
                                        parts = attr_path.split('.')
                                        target = module
                                        for p in parts:
                                            target = getattr(target, p)

                                        tmp = target.weight
                                        delattr(target, 'weight')
                                        del tmp
                                        module.weight_injection_modules.append(target)
                    elif parts[-1] == 'output_positions':
                        # Calculate required shared memory size for CUDA kernel
                        setattr(
                            module,
                            'shared_mem_size',
                            threads_per_block[0] * 4 + 4 + (
                                module.output_positions.view(torch.uint32).numpy()[1:] - \
                                    module.output_positions.view(torch.uint32).numpy()[:-1]
                            ).max().item() * 2
                        )
    
    return model


def get_no_split_classes(model, pattern_dict):
    """
    Find model layer classes that should not be split across devices.
    
    This is crucial for DFloat11 model sharding to ensure compressed modules
    stay on the same device as their decompression buffers.
    
    Args:
        model: The PyTorch model
        pattern_dict: Dictionary mapping regex patterns to submodule lists
        
    Returns:
        List of class names that should not be split across devices
    """
    no_split_classes = []
    for pattern in pattern_dict:
        for full_name, sub_module in model.named_modules():
            if re.fullmatch(pattern, full_name):
                class_name = sub_module.__class__.__name__
                if class_name not in no_split_classes:
                    no_split_classes.append(class_name)

    return no_split_classes


class DFloat11Model:
    """
    Wrapper class for loading and using models with DFloat11 compressed weights.
    DFloat11 is a custom 11-bit floating point format that provides memory efficiency
    while maintaining numerical accuracy for LLM weights.
    """
    @classmethod
    def from_pretrained(
        cls,
        dfloat11_model_name_or_path: str,
        device: Optional[str] = None,
        device_map: str = 'auto',
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        bfloat16_model = None,
        **kwargs,
    ):
        """
        Load a model with DFloat11 compressed weights from local path or Hugging Face Hub.
        
        Args:
            dfloat11_model_name_or_path: Local path or HF Hub model name
            device: Target device for the model
            device_map: Strategy for distributing model across devices
            max_memory: Maximum memory allocation per device
            bfloat16_model: Optional pre-initialized model to load weights into
            **kwargs: Additional arguments passed to AutoModelForCausalLM.from_config
            
        Returns:
            Model with DFloat11 compressed weights configured for on-the-fly decompression
        """
        # Resolve model path, downloading from HF Hub if needed
        if os.path.exists(dfloat11_model_name_or_path):
            dfloat11_model_path = dfloat11_model_name_or_path
        else:
            dfloat11_model_path = dfloat11_model_name_or_path.replace('/', '__')
            if not os.path.exists(dfloat11_model_path):
                snapshot_download(dfloat11_model_name_or_path, local_dir=dfloat11_model_path)

        # Load model configuration
        config = AutoConfig.from_pretrained(dfloat11_model_path)
        if bfloat16_model:
            model = bfloat16_model
        else:
            # Initialize model without loading weights
            with no_init_weights():
                model = AutoModelForCausalLM.from_config(
                    config, torch_dtype=torch.bfloat16, **kwargs,
                )
                model.tie_weights()
                model.eval()

            # Try to load generation config if available
            try:
                generation_config = GenerationConfig.from_pretrained(dfloat11_model_path)
                model.generation_config = generation_config
            except Exception as e:
                pass

        # Verify model has DFloat11 configuration
        assert hasattr(config, 'dfloat11_config')
        dfloat11_config = config.dfloat11_config

        # Load compressed weights and configure decompression
        load_and_replace_tensors(model, dfloat11_model_path, dfloat11_config)

        # Calculate and report model size
        model_bytes = 0
        for param in model.state_dict().values():
            if param.dtype in [torch.uint8, torch.int8]:
                model_bytes += param.numel()
            elif param.dtype in [torch.float16, torch.bfloat16, torch.int16, torch.uint16]:
                model_bytes += param.numel() * 2
            elif param.dtype in [torch.float32, torch.int32, torch.uint32]:
                model_bytes += param.numel() * 4
            elif param.dtype in [torch.float64, torch.int64, torch.uint64]:
                model_bytes += param.numel() * 8
            else:
                print(f'Unrecognized parameter data type {param.dtype}.', file=stderr)

        print(f"Total model size: {model_bytes / 1e9:0.4f} GB", file=stderr)

        # Move model to specified device or distribute across multiple devices
        if device:
            model = model.to(device)
        else:
            assert device_map == 'auto', "device_map should be 'auto' if no specific device is provided."
            # Identify modules that must remain on same device for decompression
            no_split_classes = get_no_split_classes(model, dfloat11_config['pattern_dict'])
            max_memory = get_balanced_memory(model, max_memory=max_memory, no_split_module_classes=no_split_classes)
            device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_classes)
            model = dispatch_model(model, device_map)

        # Warn if model is not fully on GPU
        if any(param.device.type == 'cpu' for param in model.parameters()):
            print("Warning: Some model layers are on CPU. For inference, ensure the model is fully loaded onto CUDA-compatible GPUs.", file=stderr)

        return model




if __name__ == "__main__":
    from transformers import AutoTokenizer
    

    
    df11_path = "./llama3.1-8b-it-df11"
    model = DFloat11Model.from_pretrained(
        df11_path,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(df11_path)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'


    inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)

    with torch.no_grad():
        # outputs = model.generate(**inputs, max_length=50, do_sample=True, top_k=50, top_p=0.95)
        outputs = model.generate(**inputs, max_new_tokens = 50, do_sample=False, top_k=None, top_p=None)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

        # Hello, how are you? I am doing well, thanks for asking. I have been busy with work and other things, but I always make time for my blog. I have been thinking about what to write about next, and I have a few ideas. I was thinking about