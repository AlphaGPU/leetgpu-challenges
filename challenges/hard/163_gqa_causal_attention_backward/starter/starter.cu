#include <cuda_runtime.h>

// Q, K, V, O, dO, lse, dQ, dK, dV are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* O,
                      const float* dO, const float* lse, float* dQ, float* dK, float* dV,
                      int num_q_heads, int num_kv_heads, int seq_len, int head_dim) {}
