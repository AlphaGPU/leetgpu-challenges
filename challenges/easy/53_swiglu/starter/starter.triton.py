import torch
import triton
import triton.language as tl

@triton.jit
def swiglu(
    input, output, N, BLOCK_SIZE: tl.constexpr
):
    pass

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N // 2, BLOCK_SIZE),)
    swiglu[grid](
        input, output, N, BLOCK_SIZE=BLOCK_SIZE
    )
