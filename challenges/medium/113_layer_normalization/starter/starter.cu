#include <cuda_runtime.h>

// input, weight, bias, output are device pointers
extern "C" void solve(const float* input, const float* weight, const float* bias, float* output,
                      int N, int C, float eps) {}
