#include <cuda_runtime.h>

// image, output are device pointers
extern "C" void solve(const float* image, float* output, int H, int W, int H_out, int W_out) {}
