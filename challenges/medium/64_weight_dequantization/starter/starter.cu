#include <cuda_runtime.h>

// X, S, Y are device pointers
extern "C" void solve(float* X, float* S, float* Y, int BLOCK_SIZE) {}