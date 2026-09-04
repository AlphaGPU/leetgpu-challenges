#include <cuda_runtime.h>

// x, c, output, weights are device pointers
extern "C" void solve(const float* x, const float* c, float* output, const float* weights,
                      int batch_size, int seq_len) {}
