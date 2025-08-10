import torch
import triton
import triton.language as tl

@triton.jit
def matrix_transpose_kernel(
    input, output,
    rows, cols,
    stride_ir, stride_ic,  
    stride_or, stride_oc
):
    pass

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1  
    stride_or, stride_oc = rows, 1
    
    grid = (rows, cols)
    matrix_transpose_kernel[grid](
        input, output,
        rows, cols,
        stride_ir, stride_ic,
        stride_or, stride_oc
    ) 