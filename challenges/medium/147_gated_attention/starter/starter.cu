#include <cuda_runtime.h>

// Q, K, V, G, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* G, float* output,
                      int M, int d) {}
