#include <cuda_runtime.h>

// x, residual, weight, out are device pointers
extern "C" void solve(const float* x, float* residual, const float* weight, float* out, int N,
                      int C, float eps) {}
