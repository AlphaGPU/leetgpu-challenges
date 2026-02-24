#include <cuda_runtime.h>

// x, output, ln1_weight, ln1_bias, W_qkv, b_qkv, W_attn_proj, b_attn_proj, ln2_weight, ln2_bias, W_fc, b_fc, W_proj, b_proj are device pointers
extern "C" void solve(const float* x, float* output, const float* ln1_weight,
                      const float* ln1_bias, const float* W_qkv, const float* b_qkv,
                      const float* W_attn_proj, const float* b_attn_proj,
                      const float* ln2_weight, const float* ln2_bias, const float* W_fc,
                      const float* b_fc, const float* W_proj, const float* b_proj,
                      int seq_len, int d_model, int n_heads, int ffn_dim) {}
