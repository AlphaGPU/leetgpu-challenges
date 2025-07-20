# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

# a_ptr, b_ptr, c_ptr are raw device pointers
def solve(a_ptr: int, b_ptr: int, c_ptr: int, BATCH: int, M: int, N: int, K: int):
    pass