import cutlass
import cutlass.cute as cute


# q, k, v, output are tensors on the GPU
@cute.jit
def solve(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    output: cute.Tensor,
    B: cute.Int32,
    H: cute.Int32,
    H_KV: cute.Int32,
    S: cute.Int32,
    D: cute.Int32,
):
    pass
