#include <cuda_runtime.h>

// x, weight, dy, dx, dweight are device pointers
extern "C" void solve(const float* x, const float* weight, const float* dy, float* dx,
                      float* dweight, int N, int C, float eps) {}
