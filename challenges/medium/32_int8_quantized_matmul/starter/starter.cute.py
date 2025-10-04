import cutlass
import cutlass.cute as cute

# A, B, C are tensors on the GPU
@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, M: cute.Int32, N: cute.Int32, K: cute.Int32, scale_A: cute.Float32, scale_B: cute.Float32, scale_C: cute.Float32, zero_point_A: cute.Int32, zero_point_B: cute.Int32, zero_point_C: cute.Int32):
    pass
