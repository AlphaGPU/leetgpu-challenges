#include <cuda_runtime.h>

// image, output are device pointers
extern "C" void solve(const float* image, float* output, int H, int W, float spatial_sigma,
                      float range_sigma, int radius) {}
