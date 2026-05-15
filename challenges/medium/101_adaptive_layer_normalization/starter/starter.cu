#include <cuda_runtime.h>

// X, scale, shift, output are device pointers
extern "C" void solve(const float* X, const float* scale, const float* shift, float* output, int B,
                      int N, int D) {}
