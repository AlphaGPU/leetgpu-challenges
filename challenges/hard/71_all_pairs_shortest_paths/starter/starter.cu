#include <cuda_runtime.h>

// dist, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* dist, float* output, int N) {}
