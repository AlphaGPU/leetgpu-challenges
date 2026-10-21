#include <cuda_runtime.h>

// Q, K, V, X, W_g, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* X,
                      const float* W_g, float* output, int M, int d_model, int h) {}
