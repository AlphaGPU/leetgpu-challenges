#include <cuda_runtime.h>

// teacher_logits, student_logits, kl are device pointers
extern "C" void solve(const float* teacher_logits, const float* student_logits, float* kl, int N,
                      int V, float temperature) {}
