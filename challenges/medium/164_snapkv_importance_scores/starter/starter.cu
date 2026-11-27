#include <cuda_runtime.h>

// Q, K, scores are device pointers
extern "C" void solve(const float* Q, const float* K, float* scores, int num_q_heads,
                      int num_kv_heads, int window, int seq_len, int head_dim) {}
