#include <cuda_runtime.h>

// q, k, v, output are device pointers
extern "C" void solve(const float* q, const float* k, const float* v, float* output, int B, int H,
                      int H_KV, int S, int D) {}
