# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

# A_ptr, x_ptr, y_ptr are raw device pointers
def solve(A_ptr: int, x_ptr: int, y_ptr: int, M: int, N: int, nnz: int):
    pass
