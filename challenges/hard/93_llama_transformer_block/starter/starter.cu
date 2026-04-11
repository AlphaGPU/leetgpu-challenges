#include <cuda_runtime.h>

// x, output, weights, cos, sin are device pointers
extern "C" void solve(const float* x, float* output, const float* weights, const float* cos,
                      const float* sin, int seq_len) {}
