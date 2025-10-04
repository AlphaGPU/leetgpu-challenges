import cutlass
import cutlass.cute as cute

# predictions, targets, mse are tensors on the GPU
@cute.jit
def solve(predictions: cute.Tensor, targets: cute.Tensor, mse: cute.Tensor, N: cute.Int32):
    pass
