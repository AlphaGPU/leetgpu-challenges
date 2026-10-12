#include <cuda_runtime.h>

// logits, output are device pointers
extern "C" void solve(const float* logits, int* output, int B, int V) {}
