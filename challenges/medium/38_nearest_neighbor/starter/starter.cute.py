import cutlass
import cutlass.cute as cute

# points, indices are tensors on the GPU
@cute.jit
def solve(points: cute.Tensor, indices: cute.Tensor, N: cute.Int32):
    pass
