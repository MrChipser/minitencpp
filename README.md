# MiniTenCpp

A small C++ tensor library currently under development.

## Current Features

- Tensor construction from a shape
- Row-major strides
- Mutable and const element access with `operator()`
- Shape, data, stride, size, size(axis) and rank getters
- Negative-dimension and size overflow validation
- Bounds and rank checking
- GoogleTest-based unit tests

## Build

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
