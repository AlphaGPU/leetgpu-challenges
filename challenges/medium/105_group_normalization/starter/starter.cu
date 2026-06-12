#include <cuda_runtime.h>

// X, gamma, beta, Y are device pointers
extern "C" void solve(const float* X, const float* gamma, const float* beta, float* Y, int N, int C,
                      int H, int W, int G, float eps) {}
