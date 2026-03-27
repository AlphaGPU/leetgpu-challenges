#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// x, w_q, scales, y are device pointers
extern "C" void solve(const __half* x, const uint8_t* w_q, const __half* scales, __half* y, int M,
                      int N, int K, int group_size) {}
