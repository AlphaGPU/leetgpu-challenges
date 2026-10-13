#include <cuda_runtime.h>

// X, W, targets, loss are device pointers
extern "C" void solve(const float* X, const float* W, const int* targets, float* loss, int N, int D,
                      int V) {}
