import cutlass
import cutlass.cute as cute

# Q, K, V, output are tensors on the GPU
@cute.jit
def solve(Q: cute.Tensor, K: cute.Tensor, V: cute.Tensor, output: cute.Tensor, N: cute.Int32, d_model: cute.Int32, h: cute.Int32):
    pass
