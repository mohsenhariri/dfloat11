# This is using V2 (Tensor Manager)

# DECODE_SO_NAME := dfloat11_decode_v2.so
CUDA_CU_SOURCE_FILE := decode.cu
CPP_WRAPPER_SOURCE_FILE := decode_wrapper.cpp
SETUP_PY := setup.py
OUTPUT_DIR := .
BUILD_DIR := build

.PHONY: all infer-df-v2 build-extension-v2 clean infer-df-jit-v2

all: infer-df-v2

build-extension-v2: $(SETUP_PY) $(CUDA_CU_SOURCE_FILE) $(CPP_WRAPPER_SOURCE_FILE)
	python $(SETUP_PY) build_ext --inplace
	@echo "Success"

# inference with the precompiled extension
infer-df-v2: build-extension-v2
	CUDA_VISIBLE_DEVICES=0 python inference_precompile_v2.py

# inference with the JIT compiled version (the current PyPI version)
infer-df-jit-v2:
	CUDA_VISIBLE_DEVICES=0 python inference_jit_v2.py


infer-df-precompile-v2:
	CUDA_VISIBLE_DEVICES=0 python inference_precompile_v2.py


clean:
	rm -f $(OUTPUT_DIR)/dfloat11_decode_v2*.so # Remove the built .so file(s)
	rm -rf $(BUILD_DIR) # Remove the build directory
	rm -rf *.egg-info # Remove egg-info directory

