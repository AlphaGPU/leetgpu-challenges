#include <cuda_runtime.h>

// rewards, values, advantages are device pointers
extern "C" void solve(const float* rewards, const float* values, float* advantages, float gamma,
                      float lam, int B, int S) {}
