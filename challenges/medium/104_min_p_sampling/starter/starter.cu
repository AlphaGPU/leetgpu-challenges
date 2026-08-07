#include <cuda_runtime.h>

// logits, probs are device pointers
extern "C" void solve(const float* logits, float* probs, float min_p, int B, int V) {}
