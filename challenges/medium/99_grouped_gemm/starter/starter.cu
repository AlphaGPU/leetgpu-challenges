#include <cuda_runtime.h>

// A, B, group_offsets, C are device pointers
extern "C" void solve(const float* A, const float* B, const int* group_offsets, float* C, int G,
                      int M_total, int K, int N) {}
