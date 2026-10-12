import cutlass
import cutlass.cute as cute


# beam_scores, token_logprobs, new_beam_scores,
# parent_beam_indices, next_tokens are tensors on the GPU
@cute.jit
def solve(
    beam_scores: cute.Tensor,
    token_logprobs: cute.Tensor,
    new_beam_scores: cute.Tensor,
    parent_beam_indices: cute.Tensor,
    next_tokens: cute.Tensor,
    B: cute.Int32,
    K: cute.Int32,
    V: cute.Int32,
):
    pass
