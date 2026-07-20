#include <cuda_runtime.h>

// rewards, log_pi, log_pi_old, log_ref, output are device pointers
extern "C" void solve(
    const float* rewards,
    const float* log_pi,
    const float* log_pi_old,
    const float* log_ref,
    float* output,
    float clip_eps,
    float beta,
    int B,
    int G,
    int S) {}
