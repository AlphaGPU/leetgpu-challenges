import torch
import triton
import triton.language as tl

@triton.jit
def count_equal_kernel(input_ptr, output_ptr, N, K, BLOCK_SIZE: tl.constexpr):
    pass

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, K: int):
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    count_equal_kernel[grid](input, output, N, K, BLOCK_SIZE=BLOCK_SIZE)