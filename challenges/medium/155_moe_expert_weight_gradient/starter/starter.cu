#include <cuda_runtime.h>

// X, dY, group_offsets, dW are device pointers
extern "C" void solve(const float* X, const float* dY, const int* group_offsets, float* dW, int G,
                      int M_total, int K, int N) {}
