#include <cuda_runtime.h>

// logits, prompt_tokens, output_tokens, presence_penalty, frequency_penalty, repetition_penalty,
// output are device pointers
extern "C" void solve(const float* logits, const int* prompt_tokens, const int* output_tokens,
                      const float* presence_penalty, const float* frequency_penalty,
                      const float* repetition_penalty, float* output, int B, int V, int P, int G) {}
