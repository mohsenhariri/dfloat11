#include <torch/extension.h> 
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>    
#include <stdexcept> 
#include <string>


extern "C" __global__ void decode(
    const unsigned char* luts,
    const unsigned char* codes,
    const unsigned char* sign_mantissa,
    const unsigned int* position_offsets,
    const unsigned char* gaps,
    unsigned short* outputs,
    int n_luts, int n_bytes, int n_elements);

void dfloat11_decode_launch_wrapper(
    uintptr_t luts_ptr,
    uintptr_t encoded_exponent_ptr,
    uintptr_t sign_mantissa_ptr,
    uintptr_t output_positions_ptr,
    uintptr_t gaps_ptr,
    uintptr_t reconstructed_output_ptr,
    int n_luts,
    int n_bytes,
    int n_elements,
    int grid_dim_x,
    int block_dim_x,
    int shared_mem_bytes
)
{
    // Use c10::cuda::getCurrentCUDAStream()
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    dim3 gridDim(grid_dim_x, 1, 1); 
    dim3 blockDim(block_dim_x, 1, 1); 

    const unsigned char* p_luts = reinterpret_cast<const unsigned char*>(luts_ptr);
    const unsigned char* p_encoded_exponent = reinterpret_cast<const unsigned char*>(encoded_exponent_ptr);
    const unsigned char* p_sign_mantissa = reinterpret_cast<const unsigned char*>(sign_mantissa_ptr);
    const unsigned int* p_output_positions = reinterpret_cast<const unsigned int*>(output_positions_ptr);
    const unsigned char* p_gaps = reinterpret_cast<const unsigned char*>(gaps_ptr);
    unsigned short* p_reconstructed_output = reinterpret_cast<unsigned short*>(reconstructed_output_ptr);

    void* kernel_args[] = {
        (void*)&p_luts,
        (void*)&p_encoded_exponent,
        (void*)&p_sign_mantissa,
        (void*)&p_output_positions,
        (void*)&p_gaps,
        (void*)&p_reconstructed_output,
        (void*)&n_luts,
        (void*)&n_bytes,
        (void*)&n_elements
    };

    cudaError_t err = cudaLaunchKernel(
        (const void*)decode,
        gridDim,
        blockDim,
        kernel_args,
        shared_mem_bytes, 
        stream);

    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed in wrapper: %s\n", cudaGetErrorString(err));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DFloat11 Decode CUDA Kernel (Pybind11/Torch C++ Extension)";
    m.def(
        "decode",
        &dfloat11_decode_launch_wrapper,
        "Launches the DFloat11 Decode CUDA kernel.",
        py::arg("luts_ptr"),
        py::arg("encoded_exponent_ptr"),
        py::arg("sign_mantissa_ptr"),
        py::arg("output_positions_ptr"),
        py::arg("gaps_ptr"),
        py::arg("reconstructed_output_ptr"),
        py::arg("n_luts"),
        py::arg("n_bytes"),
        py::arg("n_elements"),
        py::arg("grid_dim_x"),
        py::arg("block_dim_x"),
        py::arg("shared_mem_bytes")
    );
}
