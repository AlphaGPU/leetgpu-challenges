#include <cuda_runtime.h>

__global__ void subarray_sum_kernel(const int* input, int* output, int N, int S, int E) {

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int S, int E) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    subarray_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, S, E);
    cudaDeviceSynchronize();
}