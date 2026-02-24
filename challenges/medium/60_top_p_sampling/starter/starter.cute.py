import cutlass
import cutlass.cute as cute


@cute.jit
def solve(
    logits: cute.Tensor,
    p: cute.Tensor,
    top_p_probs: cute.Tensor,
    vocab_size: cute.Int32,
):
    pass
