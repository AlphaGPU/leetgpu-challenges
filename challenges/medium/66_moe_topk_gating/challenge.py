import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="MoE Top-K Gating", atol=1e-05, rtol=1e-05, num_gpus=1, access_tier="free"
        )

    def reference_impl(
        self,
        logits: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        M: int,
        E: int,
        k: int,
    ):
        """
        Computes the Top-K gating for Mixture of Experts.

        For each row in logits, select the k highest values, apply softmax to them,
        and return the weights and indices.
        """
        assert logits.shape == (M, E)
        assert topk_weights.shape == (M, k)
        assert topk_indices.shape == (M, k)
        assert logits.is_cuda and topk_weights.is_cuda and topk_indices.is_cuda
        assert topk_indices.dtype == torch.int32

        # 1. TopK Selection
        # logits: (M, E) -> vals: (M, k), indices: (M, k)
        vals, indices = torch.topk(logits, k, dim=-1)

        # 2. Softmax on the top k values
        weights = torch.softmax(vals, dim=-1)

        # 3. Write output
        topk_weights.copy_(weights)
        topk_indices.copy_(indices.to(torch.int32))
