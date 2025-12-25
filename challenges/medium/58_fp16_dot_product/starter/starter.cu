#include <cuda_fp16.h>
#include <cuda_runtime.h>

// A, B, result are device pointers
extern "C" void solve(const half* A, const half* B, half* result, int N) {}
