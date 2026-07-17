import cutlass
import cutlass.cute as cute


# rewards, values, advantages are tensors on the GPU
@cute.jit
def solve(
    rewards: cute.Tensor,
    values: cute.Tensor,
    advantages: cute.Tensor,
    gamma: cute.Float32,
    lam: cute.Float32,
    B: cute.Int32,
    S: cute.Int32,
):
    pass
