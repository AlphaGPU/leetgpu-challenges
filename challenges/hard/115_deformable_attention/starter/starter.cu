#include <cuda_runtime.h>

// value, spatial_shapes, sampling_loc, attn_weight, output are device pointers
extern "C" void solve(const float* value, const int* spatial_shapes, const float* sampling_loc,
                      const float* attn_weight, float* output, int N, int S, int Q, int H, int D,
                      int L, int P) {}
