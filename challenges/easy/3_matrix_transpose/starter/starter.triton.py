# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

@triton.jit
def matrix_transpose_kernel(
    input_ptr, output_ptr,
    rows, cols,
    stride_ir, stride_ic,  
    stride_or, stride_oc
):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))
    

# input_ptr, output_ptr are raw device pointers
def solve(input_ptr: int, output_ptr: int, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1  
    stride_or, stride_oc = rows, 1
    
    grid = (rows, cols)
    matrix_transpose_kernel[grid](
        input_ptr, output_ptr,
        rows, cols,
        stride_ir, stride_ic,
        stride_or, stride_oc
    ) 