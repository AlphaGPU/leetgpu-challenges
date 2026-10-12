#include <cuda_runtime.h>

// Q, K_idx, qjl_signs, gamma, Pi, S_mat, codebook, scores are device pointers
extern "C" void solve(const float* Q, const unsigned char* K_idx, const signed char* qjl_signs,
                      const float* gamma, const float* Pi, const float* S_mat,
                      const float* codebook, float* scores, int B, int S, int D, int C) {}
