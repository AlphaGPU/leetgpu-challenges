import torch
import triton
import triton.language as tl


@triton.jit
def interleave_kernel(A_ptr, B_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pass


# A, B, output are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 256

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    interleave_kernel[grid](A, B, output, N, BLOCK_SIZE=BLOCK_SIZE)
