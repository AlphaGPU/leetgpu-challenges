# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

@triton.jit
def reverse_kernel(
    input_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))

# input_ptr is a raw device pointer
def solve(input_ptr: int, N: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(N // 2, BLOCK_SIZE)
    grid = (n_blocks,)
    
    reverse_kernel[grid](
        input_ptr,
        N,
        BLOCK_SIZE
    ) 