import cutlass
import cutlass.cute as cute


# Q, K, V, G, output are tensors on the GPU
@cute.jit
def solve(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    G: cute.Tensor,
    output: cute.Tensor,
    M: int,
    d: int,
):
    pass
