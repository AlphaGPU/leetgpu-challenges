import cutlass
import cutlass.cute as cute

# input, histogram are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, histogram: cute.Tensor, N: cute.Int32, num_bins: cute.Int32):
    pass
