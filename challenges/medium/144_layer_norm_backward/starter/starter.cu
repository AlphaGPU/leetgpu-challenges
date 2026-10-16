#include <cuda_runtime.h>

// x, gamma, mean, rstd, dy, dx, dgamma, dbeta are device pointers
extern "C" void solve(const float* x, const float* gamma, const float* mean, const float* rstd,
                      const float* dy, float* dx, float* dgamma, float* dbeta, int N, int C) {}
