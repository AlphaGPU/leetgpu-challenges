import torch
import triton
import triton.language as tl

@triton.jit
def radix_sort_kernel(
    input, output, N
):
    input = input.to(tl.pointer_type(tl.uint32))
    output = output.to(tl.pointer_type(tl.uint32))

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    pass