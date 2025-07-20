# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl


# a_ptr, b_ptr are raw device pointers
def solve(a_ptr: int, b_ptr: int, N: int):
    pass 