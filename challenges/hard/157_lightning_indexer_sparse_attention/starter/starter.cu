#include <cuda_runtime.h>

// q, k_cache, v_cache, idx_q, idx_k, idx_w, output are device pointers
extern "C" void solve(const float* q, const float* k_cache, const float* v_cache,
                      const int8_t* idx_q, const int8_t* idx_k, const int8_t* idx_w, float* output,
                      int B, int S, int H, int D, int H_I, int D_I, int topk) {}
