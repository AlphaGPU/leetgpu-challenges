import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input, output,
    N,
    BLOCK_SIZE: tl.constexpr
):
    input = input.to(tl.pointer_type(tl.float32))
    output = output.to(tl.pointer_type(tl.float32))

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    pass