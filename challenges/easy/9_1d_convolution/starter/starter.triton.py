# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    input_ptr, kernel_ptr, output_ptr,
    input_size, kernel_size,
    BLOCK_SIZE: tl.constexpr
):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    kernel_ptr = kernel_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))

# input_ptr, kernel_ptr, output_ptr are raw device pointers
def solve(input_ptr: int, kernel_ptr: int, output_ptr: int, input_size: int, kernel_size: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE)
    grid = (n_blocks,)
    
    conv1d_kernel[grid](
        input_ptr, kernel_ptr, output_ptr,
        input_size, kernel_size,
        BLOCK_SIZE
    )