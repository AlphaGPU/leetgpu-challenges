import cutlass
import cutlass.cute as cute


# draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens are tensors on the GPU
@cute.jit
def solve(
    draft_tokens: cute.Tensor,
    draft_probs: cute.Tensor,
    target_probs: cute.Tensor,
    uniform_samples: cute.Tensor,
    output_tokens: cute.Tensor,
    B: cute.Int32,
    T: cute.Int32,
    V: cute.Int32,
):
    pass
