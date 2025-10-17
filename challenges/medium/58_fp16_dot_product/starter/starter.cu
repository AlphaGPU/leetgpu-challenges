#include <cuda_runtime.h>
#include <cuda_fp16.h>

// A, B, result are device pointers
extern "C" void solve(const __half* A, const __half* B, __half* result, int N) {

}
