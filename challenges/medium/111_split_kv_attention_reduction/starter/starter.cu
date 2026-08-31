#include <cuda_runtime.h>

// partial_out, partial_lse, output are device pointers
extern "C" void solve(const float* partial_out, const float* partial_lse, float* output,
                      int num_splits, int num_heads, int head_dim) {}
