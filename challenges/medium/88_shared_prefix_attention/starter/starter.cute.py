import cutlass
import cutlass.cute as cute


# Q, K_prefix, V_prefix, K_suffix, V_suffix, output are tensors on the GPU
@cute.jit
def solve(
    Q: cute.Tensor,
    K_prefix: cute.Tensor,
    V_prefix: cute.Tensor,
    K_suffix: cute.Tensor,
    V_suffix: cute.Tensor,
    output: cute.Tensor,
    batch_size: cute.Int32,
    num_heads: cute.Int32,
    prefix_len: cute.Int32,
    suffix_len: cute.Int32,
    head_dim: cute.Int32,
):
    pass
