import cutlass
import cutlass.cute as cute


# Q, K, V, dO, dQ, dK, dV are tensors on the GPU
@cute.jit
def solve(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    dO: cute.Tensor,
    dQ: cute.Tensor,
    dK: cute.Tensor,
    dV: cute.Tensor,
    M: cute.Int32,
    N: cute.Int32,
    d: cute.Int32,
):
    pass
