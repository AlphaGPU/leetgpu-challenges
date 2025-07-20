# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

# input_ptr, output_ptr are raw device pointers
def solve(input_ptr: int, output_ptr: int, N: int, P: int):
    pass 