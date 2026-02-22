#include <cuda_runtime.h>

// positions, masses, accelerations are device pointers
extern "C" void solve(const float* positions, const float* masses, float* accelerations, int N) {}
