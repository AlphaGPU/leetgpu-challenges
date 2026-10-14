#include <cuda_runtime.h>

// Q, K_cache, V_cache, K_tree, V_tree, parents, output are device pointers
extern "C" void solve(const float* Q, const float* K_cache, const float* V_cache,
                      const float* K_tree, const float* V_tree, const int* parents, float* output,
                      int P, int T, int H, int D) {}
