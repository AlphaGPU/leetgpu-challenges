import cutlass
import cutlass.cute as cute


# Q, K, V, q_weight, k_weight, output are tensors on the GPU
@cute.jit
def solve(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    q_weight: cute.Tensor,
    k_weight: cute.Tensor,
    output: cute.Tensor,
    N: cute.Int32,
    d_model: cute.Int32,
    h: cute.Int32,
    eps: cute.Float32,
):
    pass
