#include <cuda_runtime.h>

// Q, K, V, alpha, beta, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* alpha,
                      const float* beta, float* output, int batch, int seq_len, int d) {}
