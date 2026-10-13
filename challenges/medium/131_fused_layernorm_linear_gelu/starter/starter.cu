#include <cuda_runtime.h>

// x, gamma, beta, W, bias, y are device pointers
extern "C" void solve(const float* x, const float* gamma, const float* beta, const float* W,
                      const float* bias, float* y, int M, int K, int N, float eps) {}
