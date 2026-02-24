#include <cuda_runtime.h>

extern "C" void solve(const float* logits, const float* p, float* top_p_probs,
                      int vocab_size) {}
