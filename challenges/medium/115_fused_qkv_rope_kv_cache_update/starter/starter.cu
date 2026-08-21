#include <cuda_runtime.h>

// x, W_qkv, cos_sin_cache, positions, K_cache, V_cache, Q_out are device pointers
extern "C" void solve(const float* x, const float* W_qkv, const float* cos_sin_cache,
                      const int* positions, float* K_cache, float* V_cache, float* Q_out, int B,
                      int d_model, int H_q, int H_kv, int D, int S_max) {}
