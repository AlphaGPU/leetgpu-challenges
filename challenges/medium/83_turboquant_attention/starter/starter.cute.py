import cutlass
import cutlass.cute as cute


# Q, K_idx, Pi, codebook, scores are tensors on the GPU
@cute.jit
def solve(
    Q: cute.Tensor,
    K_idx: cute.Tensor,
    Pi: cute.Tensor,
    codebook: cute.Tensor,
    scores: cute.Tensor,
    B: cute.Int32,
    S: cute.Int32,
    D: cute.Int32,
    C: cute.Int32,
):
    pass
