import torch
import triton
import triton.language as tl


@triton.jit
def hamming_distance_kernel(codes, output, query, n_elements, BLOCK_SIZE: tl.constexpr):
    pass


# codes, output are tensors on the GPU
def solve(codes: torch.Tensor, output: torch.Tensor, query: int, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    hamming_distance_kernel[grid](codes, output, query, N, BLOCK_SIZE)
