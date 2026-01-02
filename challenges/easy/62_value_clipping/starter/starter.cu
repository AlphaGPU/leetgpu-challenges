#include <cuda_runtime.h>

__global__ void clip_kernel(const float* input, float* output, int N, float lo, float hi) {}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, float lo, float hi) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, lo, hi);
    cudaDeviceSynchronize();
}
