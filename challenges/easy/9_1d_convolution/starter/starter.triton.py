import torch
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    input, kernel, output,
    input_size, kernel_size,
    BLOCK_SIZE: tl.constexpr
):
    pass

# input, kernel, output are tensors on the GPU
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_size: int, kernel_size: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE)
    grid = (n_blocks,)
    
    conv1d_kernel[grid](
        input, kernel, output,
        input_size, kernel_size,
        BLOCK_SIZE
    )