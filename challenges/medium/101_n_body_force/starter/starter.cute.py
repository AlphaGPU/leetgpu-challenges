import cutlass
import cutlass.cute as cute


# positions, masses, forces are tensors on the GPU
@cute.jit
def solve(positions: cute.Tensor, masses: cute.Tensor, forces: cute.Tensor, N: cute.Uint32):
    pass
