import cutlass
import cutlass.cute as cute

# X, y, beta are tensors on the GPU
@cute.jit
def solve(X: cute.Tensor, y: cute.Tensor, beta: cute.Tensor, n_samples: cute.Int32, n_features: cute.Int32):
    pass
