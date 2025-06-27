from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup_dir = os.path.dirname(os.path.realpath(__file__))
cuda_source_file = os.path.join(
    setup_dir, "decode.cu"
)  # path to the decoder (to be compiled with nvcc)
wrapper_source_file = os.path.join(
    setup_dir, "decode_wrapper.cpp"
)  # path to the wrapper

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-gencode=arch=compute_52,code=sm_52",  # Maxwell
    "-gencode=arch=compute_60,code=sm_60",  # Pascal
    "-gencode=arch=compute_61,code=sm_61",
    "-gencode=arch=compute_70,code=sm_70",  # Volta: checked
    "-gencode=arch=compute_75,code=sm_75",  # Turing
    "-gencode=arch=compute_80,code=sm_80",  # Ampere / A100: checked on aisct
    "-gencode=arch=compute_86,code=sm_86",  # Ampere (RTX 30xx) checked on hpc gpu cluster
    "-gencode=arch=compute_90,code=sm_90",  # Hopper / H200 checked on aiscii
    "-gencode=arch=compute_90,code=compute_90",
]


setup(
    name="dfloat11_decode_v2",
    ext_modules=[
        CUDAExtension(
            name="dfloat11_decode_v2",
            sources=[wrapper_source_file, cuda_source_file],
            extra_compile_args={
                "cxx": ["-g", "-O3"],
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
