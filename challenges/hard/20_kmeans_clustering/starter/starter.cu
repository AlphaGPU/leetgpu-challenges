#include <cuda_runtime.h>

// data_x, data_y, labels, initial_centroid_x, initial_centroid_y,
// final_centroid_x, final_centroid_y are device pointers 
extern "C" void solve(const float* data_x, const float* data_y, int* labels,
           float* initial_centroid_x, float* initial_centroid_y,
           float* final_centroid_x, float* final_centroid_y,
           int sample_size, int k, int max_iterations) {

}
