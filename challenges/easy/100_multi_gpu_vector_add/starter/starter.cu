#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>

// A pre-generated NCCL unique id is written by the runner at LEETGPU_NCCL_ID_FILE
// before solve() is called, so every rank can simply read it.
static ncclComm_t get_comm() {
    static ncclComm_t comm = nullptr;
    if (comm)
        return comm;

    int rank = atoi(getenv("RANK"));
    int world_size = atoi(getenv("WORLD_SIZE"));
    const char* id_path = getenv("LEETGPU_NCCL_ID_FILE");

    ncclUniqueId id;
    FILE* f = fopen(id_path, "rb");
    fread(&id, sizeof(id), 1, f);
    fclose(f);

    ncclCommInitRank(&comm, world_size, id, rank);
    return comm;
}

__global__ void vector_add_slice(const float* A, const float* B, float* C, int lo, int hi) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = lo + tid;
    if (idx < hi)
        C[idx] = A[idx] + B[idx];
}

// A, B, C are device pointers (inputs replicated across all ranks).
// Each rank must leave the full result in C.
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int rank = atoi(getenv("RANK"));
    int world_size = atoi(getenv("WORLD_SIZE"));
    ncclComm_t comm = get_comm();

    int chunk = (N + world_size - 1) / world_size;
    int lo = rank * chunk;
    int hi = lo + chunk > N ? N : lo + chunk;

    cudaMemset(C, 0, N * sizeof(float));

    int local = hi - lo;
    if (local > 0) {
        int block = 256;
        int grid = (local + block - 1) / block;
        vector_add_slice<<<grid, block>>>(A, B, C, lo, hi);
    }

    ncclAllReduce((const void*)C, (void*)C, N, ncclFloat, ncclSum, comm, 0);
    cudaDeviceSynchronize();
}
