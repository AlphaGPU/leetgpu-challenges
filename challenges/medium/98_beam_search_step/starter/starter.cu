#include <cuda_runtime.h>

// beam_scores, token_logprobs, new_beam_scores, parent_beam_indices, next_tokens are device
// pointers
extern "C" void solve(const float* beam_scores, const float* token_logprobs, float* new_beam_scores,
                      int* parent_beam_indices, int* next_tokens, int B, int K, int V) {}
