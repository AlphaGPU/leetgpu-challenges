# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

# X_ptr, y_ptr, beta_ptr are raw device pointers
def solve(X_ptr: int, y_ptr: int, beta_ptr: int, n_samples: int, n_features: int):
    pass
