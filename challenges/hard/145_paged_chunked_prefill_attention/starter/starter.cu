#include <cuda_runtime.h>

// Q, K_cache, V_cache, block_table, context_lens, cu_seqlens_q, output are device pointers
extern "C" void solve(const float* Q, const float* K_cache, const float* V_cache,
                      const int* block_table, const int* context_lens, const int* cu_seqlens_q,
                      float* output, int num_seqs, int total_q, int num_q_heads, int num_kv_heads,
                      int head_dim, int block_size, int max_blocks_per_seq) {}
