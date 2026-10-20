#include <cuda_runtime.h>

// Q, K, V, lambdas, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* lambdas,
                      float* output, int seq_len, int num_heads, int head_dim, float lambda_init) {}
