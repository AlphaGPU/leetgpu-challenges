#include <cuda_runtime.h>

// x, weights, cos_sin_cache, positions, K_cache, V_cache, output are device pointers
extern "C" void solve(const float* x, const float* weights, const float* cos_sin_cache,
                      const int* positions, float* K_cache, float* V_cache, float* output, int B,
                      int d_model, int H_q, int H_kv, int D, int d_ffn, int S_max) {}
