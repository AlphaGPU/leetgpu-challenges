#include <cuda_runtime.h>
#include <stdint.h>

// x, w_values, w_meta, y are device pointers
extern "C" void solve(const float* x, const float* w_values, const uint8_t* w_meta, float* y, int M,
                      int N, int K) {}
