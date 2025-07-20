# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

# data_ptr and output_ptr are raw device pointers
def solve(data_ptr: int, output_ptr: int, n: int):
    pass
