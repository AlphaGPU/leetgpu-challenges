#include <cuda_runtime.h>

// Q, K, V, q_weight, k_weight, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* q_weight,
                      const float* k_weight, float* output, int N, int d_model, int h, float eps) {}
