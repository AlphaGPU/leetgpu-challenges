# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

# a_ptr, b_ptr, c_ptr are raw device pointers
def solve(a_ptr: int, b_ptr: int, c_ptr: int, M: int, N: int, K: int, scale_A: float, scale_B: float, scale_C: float, zero_point_A: int, zero_point_B: int, zero_point_C: int):
    pass 