# AGENTS.md

## Build

```bash
cmake -B build -S .    # configure
cmake --build build    # build cppx executable
./build/cppx          # run
```

Debug build: `cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug .`

## Project Structure

- `src/main.cpp` - entry point
- `src/funcs.cpp` / `src/funcs.h` - C++ utilities
- `src/k0.cu` - CUDA kernel with Tensor Core MMA (`mma.sync.aligned.m16n8k8`)

## Requirements

- CMake 3.18+
- NVCC with CUDA Toolkit (kernel uses tensor core intrinsics)
- C++14

## Key Notes

- Kernel uses TF32 tensor core MMA (16x8x8 tile), validates results against CPU reference within 1% tolerance
- Code has intentional compile errors (see `lol<float> = new float();` on line 13 of main.cpp - missing template instantiation)