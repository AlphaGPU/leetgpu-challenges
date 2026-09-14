import torch


# beam_scores, token_logprobs, new_beam_scores,
# parent_beam_indices, next_tokens are tensors on the GPU
def solve(
    beam_scores: torch.Tensor,
    token_logprobs: torch.Tensor,
    new_beam_scores: torch.Tensor,
    parent_beam_indices: torch.Tensor,
    next_tokens: torch.Tensor,
    B: int,
    K: int,
    V: int,
):
    pass
