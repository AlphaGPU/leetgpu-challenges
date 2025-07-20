#include <cuda_runtime.h>

// A, B, C are device pointers
extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K, float scale_A, float scale_B, float scale_C, int zero_point_A, int zero_point_B, int zero_point_C) {

} 