import cutlass
import cutlass.cute as cute


# q, k_cache, v_cache, idx_q, idx_k, idx_w, output are tensors on the GPU
@cute.jit
def solve(
    q: cute.Tensor,
    k_cache: cute.Tensor,
    v_cache: cute.Tensor,
    idx_q: cute.Tensor,
    idx_k: cute.Tensor,
    idx_w: cute.Tensor,
    output: cute.Tensor,
    B: cute.Int32,
    S: cute.Int32,
    H: cute.Int32,
    D: cute.Int32,
    H_I: cute.Int32,
    D_I: cute.Int32,
    topk: cute.Int32,
):
    pass
