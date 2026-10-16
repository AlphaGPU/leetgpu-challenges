#include <cuda_runtime.h>

// x, residual, gamma, beta, y, residual_out, mean, rstd are device pointers
extern "C" void solve(const float* x, const float* residual, const float* gamma, const float* beta,
                      float* y, float* residual_out, float* mean, float* rstd, int N, int C,
                      float eps) {}
