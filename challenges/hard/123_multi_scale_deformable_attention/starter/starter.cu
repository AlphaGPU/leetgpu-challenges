#include <cuda_runtime.h>

// value, spatial_shapes, sampling_loc, attn_weight, output are device pointers
extern "C" void solve(const float* value, const int* spatial_shapes, const float* sampling_loc,
                      const float* attn_weight, float* output, int num_queries, int num_heads,
                      int head_dim, int num_levels, int num_points) {}
