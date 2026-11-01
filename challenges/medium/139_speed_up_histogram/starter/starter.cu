#include <cuda_runtime.h>

__global__ void histogram_kernel(const int* input, int* histogram, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        atomicAdd(&histogram[input[idx]], 1);
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemset(histogram, 0, (size_t)num_bins * sizeof(int));
    histogram_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, histogram, N);
    cudaDeviceSynchronize();
}
