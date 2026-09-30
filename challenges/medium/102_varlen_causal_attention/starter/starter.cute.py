import cutlass
import cutlass.cute as cute


# Q, K, V, cu_seqlens, output are tensors on the GPU
@cute.jit
def solve(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    cu_seqlens: cute.Tensor,
    output: cute.Tensor,
    T: cute.Int32,
    d: cute.Int32,
    S: cute.Int32,
):
    pass
