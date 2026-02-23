#include <cuda_runtime.h>

// A, out are device pointers
extern "C" void solve(const float* A, int N, float* out) {}
