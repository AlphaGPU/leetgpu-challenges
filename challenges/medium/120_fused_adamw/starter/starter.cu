#include <cuda_runtime.h>

// params, grad, m, v are device pointers
extern "C" void solve(float* params, const float* grad, float* m, float* v, int N, float lr,
                      float beta1, float beta2, float eps, float weight_decay, int t) {}
