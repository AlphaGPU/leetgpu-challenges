#include <cuda_runtime.h>

__global__ void count_3d_equal_kernel(const int* input, int* output, int N, int M, int K, int P) {

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K, int P) {
    dim3 threadsPerBlock(8, 8, 8);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                              (M + threadsPerBlock.y - 1) / threadsPerBlock.y,
                              (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    count_3d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, K, P);
    cudaDeviceSynchronize();
}