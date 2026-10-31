#include <cuda_runtime.h>

// X, W, targets, dX, dW are device pointers
extern "C" void solve(const float* X, const float* W, const int* targets, float* dX, float* dW,
                      int N, int D, int V) {}
