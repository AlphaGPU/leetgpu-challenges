#include <cuda_runtime.h>

// G, Buf, output are device pointers
extern "C" void solve(const float* G, const float* Buf, float* output, int M, int N,
                      float momentum) {}
