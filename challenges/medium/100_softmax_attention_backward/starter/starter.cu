#include <cuda_runtime.h>

// Q, K, V, dO, dQ, dK, dV are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* dO, float* dQ,
                      float* dK, float* dV, int M, int N, int d) {}
