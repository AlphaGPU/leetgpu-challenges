#include <cuda_runtime.h>

#define BLOCK 256

__global__ void row_sum_kernel(const float* input, float* output, int M, int N) {
    __shared__ float sdata[BLOCK];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* in = input + (long long)row * N;
    float acc = 0.0f;
    for (int base = 0; base < N; base += BLOCK) {
        sdata[tid] = (base + tid < N) ? in[base + tid] : 0.0f;
        __syncthreads();
        for (int s = 1; s < BLOCK; s *= 2) {
            if (tid % (2 * s) == 0)
                sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0)
            acc += sdata[0];
        __syncthreads();
    }
    if (tid == 0)
        output[row] = acc;
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int M, int N) {
    row_sum_kernel<<<M, BLOCK>>>(input, output, M, N);
    cudaDeviceSynchronize();
}
