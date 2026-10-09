#include <cuda_runtime.h>

__global__ void hamming_distance_kernel(const int* codes, int* output, int query, int N) {}

// codes, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* codes, int* output, int query, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    hamming_distance_kernel<<<blocksPerGrid, threadsPerBlock>>>(codes, output, query, N);
    cudaDeviceSynchronize();
}
