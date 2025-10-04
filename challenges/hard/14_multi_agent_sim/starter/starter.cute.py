import cutlass
import cutlass.cute as cute

# agents, agents_next are tensors on the GPU
@cute.jit
def solve(agents: cute.Tensor, agents_next: cute.Tensor, N: cute.Int32):
    pass
