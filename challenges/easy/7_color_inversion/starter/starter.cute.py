import cutlass
import cutlass.cute as cute

# image are tensors on the GPU
@cute.jit
def solve(image: cute.Tensor, width: cute.Int32, height: cute.Int32):
    pass
