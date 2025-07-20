import triton
import triton.language as tl

@triton.jit
def relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))
    
# input_ptr, output_ptr are raw device pointers
def solve(input_ptr: int, output_ptr: int, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    relu_kernel[grid](input_ptr, output_ptr, N, BLOCK_SIZE)
