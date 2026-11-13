#include <cuda_runtime.h>

// image_embeds, text_embeds, loss are device pointers
extern "C" void solve(const float* image_embeds, const float* text_embeds, float* loss, int N,
                      int D, float logit_scale) {}
