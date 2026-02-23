import cutlass
import cutlass.cute as cute


# A, out are tensors on the GPU
@cute.jit
def solve(A: cute.Tensor, N: cute.Uint32, out: cute.Tensor):
    pass
