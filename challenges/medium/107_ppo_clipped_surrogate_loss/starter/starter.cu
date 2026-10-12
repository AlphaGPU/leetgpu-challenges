#include <cuda_runtime.h>

// advantages, log_pi, log_pi_old, output are device pointers
extern "C" void solve(const float* advantages, const float* log_pi, const float* log_pi_old,
                      float* output, float clip_eps, int B, int S) {}
