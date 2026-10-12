#include <cuda_runtime.h>

// src, index, output are device pointers
extern "C" void solve(const float* src, const int* index, float* output, int N, int D, int M) {}
