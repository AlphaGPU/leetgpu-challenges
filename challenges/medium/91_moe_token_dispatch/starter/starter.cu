#include <cuda_runtime.h>

// x, expert_idx, dispatched_x, token_counts are device pointers
extern "C" void solve(const float* x, const int* expert_idx, float* dispatched_x, int* token_counts,
                      int T, int D, int E, int capacity) {}
