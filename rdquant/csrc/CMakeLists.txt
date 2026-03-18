cmake_minimum_required(VERSION 3.18)
project(rdquant_cuda LANGUAGES CXX CUDA)

# CMAKE_CUDA_ARCHITECTURES strips the 'a' suffix, but we need sm_120a
# for __CUDA_ARCH_FEAT_SM120_ALL (required by CUTLASS block-scaled MMA).
set(CMAKE_CUDA_ARCHITECTURES OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_120a,code=sm_120a")
set(CUTLASS_ROOT "/root/autodl-tmp/TensorRT-LLM-src/cpp/build-official-full-micro/_deps/cutlass-src")

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Find PyTorch
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)

# Get pybind11 include dirs
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# CUTLASS includes
include_directories(${CUTLASS_ROOT}/include)
include_directories(${CUTLASS_ROOT}/tools/util/include)
include_directories(${CUTLASS_ROOT}/examples/common)

# Build the shared library manually (avoid pybind11_add_module)
add_library(rdquant_cuda MODULE
    bindings.cpp
    w4a8.cu
    w6a8.cu
    w8a8.cu
    sf_layout.cu
    cublas_mxfp8.cu
)

target_include_directories(rdquant_cuda PRIVATE
    ${PYBIND11_INCLUDE}
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# Find torch_python for pybind11 Tensor type casters
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/lib')"
    OUTPUT_VARIABLE TORCH_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

target_link_libraries(rdquant_cuda PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_LIB_DIR}/libtorch_python.so
    Python3::Python
    CUDA::cublasLt
)

# CUDA compile options
target_compile_options(rdquant_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Xcompiler=-fPIC
        --std=c++17
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -fPIC
    >
)

# C++ standard
set_target_properties(rdquant_cuda PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    PREFIX ""
    SUFFIX ".so"
)

target_compile_definitions(rdquant_cuda PRIVATE
    CUTLASS_ARCH_SM120_SUPPORTED
    TORCH_EXTENSION_NAME=rdquant_cuda
)
