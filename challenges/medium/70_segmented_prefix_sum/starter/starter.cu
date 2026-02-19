#include <cuda_runtime.h>

// values, flags, output are device pointers
extern "C" void solve(const float* values, const int* flags, float* output, int N) {}
