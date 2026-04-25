cmake_minimum_required(VERSION 3.18)
project(cppx LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures: Hopper (sm_90a) and Blackwell (sm_120)
set(CMAKE_CUDA_ARCHITECTURES 90a 120)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(cppx
    src/main.cpp
    src/funcs.cpp
    src/k0.cu
)

target_include_directories(cppx PRIVATE
    src
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(cppx PRIVATE
    CUDA::cudart
)
