#include <cuda_runtime.h>

// chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps, output are device pointers
extern "C" void solve(const float* chosen_logps, const float* rejected_logps,
                      const float* chosen_ref_logps, const float* rejected_ref_logps, float* output,
                      float beta, int B) {}
