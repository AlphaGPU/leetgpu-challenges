#include <cuda_runtime.h>

// x, W_gate_up, output are device pointers
extern "C" void solve(const float* x, const float* W_gate_up, float* output, int M, int H, int I) {}
