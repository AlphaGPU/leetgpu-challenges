# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

@triton.jit
def invert_kernel(
    image_ptr,
    width, height,
    BLOCK_SIZE: tl.constexpr
):
    image_ptr = image_ptr.to(tl.pointer_type(tl.uint8))

# image_ptr is a raw device pointer
def solve(image_ptr: int, width: int, height: int):
    BLOCK_SIZE = 1024
    n_pixels = width * height
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    
    invert_kernel[grid](
        image_ptr,
        width, height,
        BLOCK_SIZE
    ) 