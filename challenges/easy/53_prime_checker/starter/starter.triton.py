import torch
import triton
import triton.language as tl

@triton.jit
def prime_checker_kernel(n, output_ptr, BLOCK_SIZE: tl.constexpr):
    pass

# output are tensors on the GPU
def solve(n: int, output: torch.Tensor):
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    prime_checker_kernel[grid](n, output, BLOCK_SIZE=BLOCK_SIZE)