import cutlass
import cutlass.cute as cute


# X, S, Y are tensors on the GPU
@cute.jit
def solve(X: cute.Tensor, S: cute.Tensor, Y: cute.Tensor, BLOCK_SIZE: cute.Int32):
    pass
