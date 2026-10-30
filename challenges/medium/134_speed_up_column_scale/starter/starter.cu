#include <cuda_runtime.h>

__global__ void column_scale_kernel(const float* input, const float* gamma, const float* beta,
                                    float* output, int M, int N) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)M * N;
    if (idx >= total)
        return;
    int row = (int)(idx % M);
    int col = (int)(idx / M);
    long long offset = (long long)row * N + col;
    output[offset] = input[offset] * gamma[col] + beta[col];
}

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output,
                      int M, int N) {
    long long total = (long long)M * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (int)((total + threadsPerBlock - 1) / threadsPerBlock);

    column_scale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, gamma, beta, output, M, N);
    cudaDeviceSynchronize();
}
