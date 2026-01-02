import torch
import triton
import triton.language as tl


@triton.jit
def clip_kernel(input, output, N, lo, hi, BLOCK_SIZE: tl.constexpr):
    pass


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, lo: float, hi: float):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    clip_kernel[grid](input, output, N, lo, hi, BLOCK_SIZE=BLOCK_SIZE)
