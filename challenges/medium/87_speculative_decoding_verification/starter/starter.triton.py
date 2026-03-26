import torch
import triton
import triton.language as tl


# draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens are tensors on the GPU
def solve(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    output_tokens: torch.Tensor,
    B: int,
    T: int,
    V: int,
):
    pass
