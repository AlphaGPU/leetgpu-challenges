#include <cuda_runtime.h>

__global__ void segment_reverse_kernel(const float* input, float* output, int S) {
    extern __shared__ float buf[];
    const float* in = input + (long long)blockIdx.x * S;
    float* out = output + (long long)blockIdx.x * S;
    for (int i = threadIdx.x; i < S; i += blockDim.x)
        buf[i] = in[i];
    __syncthreads();
    for (int i = threadIdx.x; i < S; i += blockDim.x)
        out[i] = buf[S - 1 - i];
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int S) {
    int segments = N / S;
    int threadsPerBlock = 32;
    size_t sharedBytes = (size_t)S * sizeof(float);

    segment_reverse_kernel<<<segments, threadsPerBlock, sharedBytes>>>(input, output, S);
    cudaDeviceSynchronize();
}
