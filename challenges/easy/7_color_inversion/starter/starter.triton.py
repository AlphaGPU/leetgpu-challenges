import torch
import triton
import triton.language as tl

@triton.jit
def invert_kernel(
    image,
    width, height,
    BLOCK_SIZE: tl.constexpr
):
    pass

# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    BLOCK_SIZE = 1024
    n_pixels = width * height
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    
    invert_kernel[grid](
        image,
        width, height,
        BLOCK_SIZE
    ) 