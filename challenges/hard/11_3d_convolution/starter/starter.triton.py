# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

# input, kernel, output are raw device pointers on the GPU
def solve(input_ptr: int, kernel_ptr: int, output_ptr: int,
          input_depth: int, input_rows: int, input_cols: int,
          kernel_depth: int, kernel_rows: int, kernel_cols: int):
    pass 