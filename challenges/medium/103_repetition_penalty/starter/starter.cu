#include <cuda_runtime.h>

// logits, input_ids are device pointers
extern "C" void solve(float* logits, const int* input_ids, float penalty, int B, int V, int T) {}
