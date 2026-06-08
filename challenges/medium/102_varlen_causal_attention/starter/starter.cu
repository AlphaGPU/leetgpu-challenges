#include <cuda_runtime.h>

// Q, K, V, cu_seqlens, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const int* cu_seqlens,
                      float* output, int T, int d, int S) {}
