#include <cuda_runtime.h>

// Q, K_prefix, V_prefix, K_suffix, V_suffix, output are device pointers
extern "C" void solve(const float* Q, const float* K_prefix, const float* V_prefix,
                      const float* K_suffix, const float* V_suffix, float* output, int batch_size,
                      int num_heads, int prefix_len, int suffix_len, int head_dim) {}
