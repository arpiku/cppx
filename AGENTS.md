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
- `src/header.h` - inline helpers (`foo`, `lol` template variable)

## Requirements

- CMake 3.18+
- NVCC with CUDA Toolkit (kernel uses tensor core intrinsics)
- C++17

## CUDA Architecture Targets

- `sm_90a` — Hopper (with acceleration structures)
- `sm_120` — Blackwell

Both architectures require CUDA toolkit support (verified on CUDA 13.2).

## Key Notes

- Kernel uses TF32 tensor core MMA (16×8×8 tile), validates results against CPU reference within 1% tolerance
- `.zed/debug.json` has preconfigured debug/release launch tasks for Zed editor
- `.gitignore` excludes `build/`, `build-debug/`, `build-release/`, `*.o`, `*.out`, `run`, `cppx`, `.cache/`, and `compile_commands.json`
