#include <cuda_runtime.h>

__global__ void prime_checker_kernel(int n, int* output) {
}

// output is device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(int n, int* output) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    prime_checker_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, output);
    cudaDeviceSynchronize(); 
}