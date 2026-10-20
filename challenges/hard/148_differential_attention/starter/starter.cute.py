import cutlass
import cutlass.cute as cute


# Q, K, V, lambdas, output are tensors on the GPU
@cute.jit
def solve(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    lambdas: cute.Tensor,
    output: cute.Tensor,
    seq_len: cute.Int32,
    num_heads: cute.Int32,
    head_dim: cute.Int32,
    lambda_init: cute.Float32,
):
    pass
