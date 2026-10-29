#include <cuda_runtime.h>

// x, dt, dt_bias, A, B, C, D, z, y are device pointers
extern "C" void solve(const float* x, const float* dt, const float* dt_bias, const float* A,
                      const float* B, const float* C, const float* D, const float* z, float* y,
                      int batch, int seq_len, int num_heads, int head_dim, int num_groups,
                      int state_dim) {}
