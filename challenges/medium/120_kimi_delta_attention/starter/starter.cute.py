import cutlass
import cutlass.cute as cute


# Q, K, V, alpha, beta, output are tensors on the GPU
@cute.jit
def solve(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    alpha: cute.Tensor,
    beta: cute.Tensor,
    output: cute.Tensor,
    batch: cute.Uint32,
    seq_len: cute.Uint32,
    d: cute.Uint32,
):
    pass
