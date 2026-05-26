#include <cuda_runtime.h>

// x, residual, weight, output are device pointers
extern "C" void solve(const float* x, const float* residual, const float* weight, float* output,
                      int M, int N, float eps) {}
