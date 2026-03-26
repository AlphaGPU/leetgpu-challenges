#include <cuda_runtime.h>

// Q, K_idx, Pi, codebook, scores are device pointers
extern "C" void solve(const float* Q, const unsigned char* K_idx, const float* Pi,
                      const float* codebook, float* scores, int B, int S, int D, int C) {}
