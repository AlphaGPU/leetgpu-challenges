#include <cuda_runtime.h>

// boxes, scores, keep are device pointers
extern "C" void solve(const float* boxes, const float* scores, int* keep, int N,
                      float iou_threshold) {}
