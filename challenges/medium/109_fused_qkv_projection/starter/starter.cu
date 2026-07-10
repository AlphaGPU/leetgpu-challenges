#include <cuda_runtime.h>

// x, W_qkv, Q, K, V are device pointers
extern "C" void solve(const float* x, const float* W_qkv, float* Q, float* K, float* V, int M,
                      int num_heads, int head_dim) {}
