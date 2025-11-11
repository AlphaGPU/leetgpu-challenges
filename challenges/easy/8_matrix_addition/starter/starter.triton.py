import torch
import triton
import triton.language as tl

@triton.jit
def matrix_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pass
   
# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):    
    BLOCK_SIZE = 1024
    n_elements = N * N
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    matrix_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)
