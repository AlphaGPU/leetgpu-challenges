#include <cuda_runtime.h>

// a, x, h are device pointers
extern "C" void solve(const float* a, const float* x, float* h, int B, int L) {}
