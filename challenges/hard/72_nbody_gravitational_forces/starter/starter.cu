#include <cuda_runtime.h>

// positions[i*3+0..2], masses[i], accelerations[i*3+0..2] are device pointers
extern "C" void solve(const float* positions, const float* masses, float* accelerations, int N) {}
