#include <cuda_runtime.h>

// x, topk_ids, topk_weights, w_gate, w_up, w_down, output are device pointers
extern "C" void solve(const float* x, const int* topk_ids, const float* topk_weights,
                      const float* w_gate, const float* w_up, const float* w_down, float* output,
                      int T, int H, int I, int E, int K) {}
