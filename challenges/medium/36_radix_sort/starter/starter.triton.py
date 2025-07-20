# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

@triton.jit
def radix_sort_kernel(
    input_ptr, output_ptr, N
):
    input_ptr = input_ptr.to(tl.pointer_type(tl.uint32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.uint32))

# input_ptr, output_ptr are raw device pointers
def solve(input_ptr: int, output_ptr: int, N: int):
    pass 