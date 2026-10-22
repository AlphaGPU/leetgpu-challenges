#include <cuda_runtime.h>

// x, gamma, W, w_scale, output are device pointers
extern "C" void solve(const float* x, const float* gamma, const int8_t* W, const float* w_scale,
                      float* output, int M, int N, int K) {}
