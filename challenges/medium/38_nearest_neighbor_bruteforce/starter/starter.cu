#include <cuda_runtime.h>

/*
 * points   – device pointer to N*3 float32  (x, y, z interleaved)
 * indices  – device pointer to N   int32   (output)
 * N        – number of points
 */
__global__ void nn_kernel(const float* points, int* indices, int N)
{
    // TODO: give each thread one (or more) points[i],
    //       compute arg-min over j ≠ i, write indices[i].
}

extern "C" void solve(const float* points, int* indices, int N)
{
    /* Launch parameters are placeholders – adjust as needed. */
    constexpr int BLOCK = 256;
    int grid = (N + BLOCK - 1) / BLOCK;
    nn_kernel<<<grid, BLOCK>>>(points, indices, N);
    cudaDeviceSynchronize();
}
