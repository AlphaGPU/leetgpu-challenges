import cutlass
import cutlass.cute as cute


# Q, K, V, output are tensors on the GPU
@cute.jit
def solve(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    output: cute.Tensor,
    seq_len: cute.Int32,
    d_model: cute.Int32,
    gamma: cute.Float32,
):
    pass
