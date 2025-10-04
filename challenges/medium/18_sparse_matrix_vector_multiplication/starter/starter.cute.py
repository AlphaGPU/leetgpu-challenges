import cutlass
import cutlass.cute as cute

# A, x, y are tensors on the GPU
@cute.jit
def solve(A: cute.Tensor, x: cute.Tensor, y: cute.Tensor, M: cute.Int32, N: cute.Int32, nnz: cute.Int32):
    pass
