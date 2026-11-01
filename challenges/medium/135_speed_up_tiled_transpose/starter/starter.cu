#include <cuda_runtime.h>

#define TILE 32

__global__ void transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE][TILE];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < cols && y < rows)
        tile[threadIdx.x][threadIdx.y] = input[(long long)y * cols + x];
    __syncthreads();
    int ox = blockIdx.y * TILE + threadIdx.x;
    int oy = blockIdx.x * TILE + threadIdx.y;
    if (ox < rows && oy < cols)
        output[(long long)oy * rows + ox] = tile[threadIdx.y][threadIdx.x];
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);

    transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
