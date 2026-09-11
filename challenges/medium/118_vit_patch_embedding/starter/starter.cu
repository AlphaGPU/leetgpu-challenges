#include <cuda_runtime.h>

// images, patch_weight, patch_bias, cls_token, pos_embed, output are device pointers
extern "C" void solve(const float* images, const float* patch_weight, const float* patch_bias,
                      const float* cls_token, const float* pos_embed, float* output, int B, int C,
                      int H, int W, int P, int D) {}
