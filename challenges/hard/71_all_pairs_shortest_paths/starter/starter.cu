#include <cuda_runtime.h>

// dist, output are device pointers (N*N floats each, row-major)
extern "C" void solve(const float* dist, float* output, int N) {}
