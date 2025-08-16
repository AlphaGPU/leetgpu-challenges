#include <cuda_runtime.h>

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output,
                      int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {
  
}