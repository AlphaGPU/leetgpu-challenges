#include <cuda_runtime.h>
#include <cuda_fp16.h>

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {

}
