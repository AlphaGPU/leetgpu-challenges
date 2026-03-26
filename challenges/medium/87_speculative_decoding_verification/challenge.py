import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Speculative Decoding Verification",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        target_probs: torch.Tensor,
        uniform_samples: torch.Tensor,
        output_tokens: torch.Tensor,
        B: int,
        T: int,
        V: int,
    ):
        assert draft_tokens.shape == (B, T)
        assert draft_probs.shape == (B, T, V)
        assert target_probs.shape == (B, T, V)
        assert uniform_samples.shape == (B, T + 1)
        assert output_tokens.shape == (B, T + 1)
        assert draft_tokens.dtype == torch.int32
        assert draft_probs.dtype == torch.float32
        assert target_probs.dtype == torch.float32
        assert uniform_samples.dtype == torch.float32
        assert output_tokens.dtype == torch.int32
        assert draft_tokens.device.type == "cuda"
        assert draft_probs.device.type == "cuda"
        assert target_probs.device.type == "cuda"
        assert uniform_samples.device.type == "cuda"
        assert output_tokens.device.type == "cuda"

        output_tokens.fill_(0)

        for b in range(B):
            for i in range(T):
                tok = int(draft_tokens[b, i].item())
                p = draft_probs[b, i, tok].item()
                q = target_probs[b, i, tok].item()
                alpha = min(1.0, q / p)

                if uniform_samples[b, i].item() < alpha:
                    output_tokens[b, i] = tok
                else:
                    adjusted = torch.clamp(target_probs[b, i] - draft_probs[b, i], min=0.0)
                    total = adjusted.sum().item()
                    if total > 0.0:
                        adjusted = adjusted / total
                    else:
                        adjusted = (
                            torch.ones(V, dtype=torch.float32, device=draft_tokens.device) / V
                        )
                    cdf = torch.cumsum(adjusted, dim=0)
                    r = float(uniform_samples[b, T].item())
                    new_tok = int(torch.searchsorted(cdf.contiguous(), r).item())
                    output_tokens[b, i] = min(new_tok, V - 1)
                    break
            else:
                cdf = torch.cumsum(target_probs[b, T - 1], dim=0)
                r = float(uniform_samples[b, T].item())
                bonus_tok = int(torch.searchsorted(cdf.contiguous(), r).item())
                output_tokens[b, T] = min(bonus_tok, V - 1)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "draft_tokens": (ctypes.POINTER(ctypes.c_int), "in"),
            "draft_probs": (ctypes.POINTER(ctypes.c_float), "in"),
            "target_probs": (ctypes.POINTER(ctypes.c_float), "in"),
            "uniform_samples": (ctypes.POINTER(ctypes.c_float), "in"),
            "output_tokens": (ctypes.POINTER(ctypes.c_int), "out"),
            "B": (ctypes.c_int, "in"),
            "T": (ctypes.c_int, "in"),
            "V": (ctypes.c_int, "in"),
        }

    def _make_sparse_probs(self, B, T, V, K, device):
        """Generate sparse probability distributions: only K tokens have nonzero probability.

        Using sparse distributions ensures that the adjusted distribution clamp(q-p, 0)
        has at most 2K nonzero entries, making CDF summation numerically exact regardless
        of summation order. This prevents floating-point sensitivity for large V.
        """
        K = min(K, V)
        flat = B * T
        # For each (b, i), sample K distinct token indices
        idx = torch.stack([torch.randperm(V, device=device)[:K] for _ in range(flat)])
        idx = idx.view(B, T, K)
        # Random weights summing to 1
        weights = torch.rand(B, T, K, device=device)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        # Scatter into full V-dimensional probability vector
        probs = torch.zeros(B, T, V, device=device)
        probs.scatter_(2, idx, weights)
        return probs, idx

    def _make_test_case(self, B, T, V, seed=42):
        torch.manual_seed(seed)
        device = "cuda"

        # K=64 active tokens per position: enough diversity while keeping the adjusted
        # distribution sparse (at most 128 nonzero entries), ensuring CDF sums are
        # independent of floating-point summation order.
        K = min(64, V)
        draft_probs, draft_idx = self._make_sparse_probs(B, T, V, K, device)
        target_probs, _ = self._make_sparse_probs(B, T, V, K, device)

        # Sample draft tokens from the active K tokens
        weights = draft_probs.gather(2, draft_idx)  # (B, T, K)
        flat_w = weights.view(B * T, K)
        chosen = torch.multinomial(flat_w, 1).view(B, T)  # index within the K tokens
        draft_tokens = draft_idx.gather(2, chosen.unsqueeze(-1)).squeeze(-1).to(torch.int32)

        uniform_samples = torch.rand(B, T + 1, device=device)
        output_tokens = torch.zeros(B, T + 1, device=device, dtype=torch.int32)

        return {
            "draft_tokens": draft_tokens,
            "draft_probs": draft_probs,
            "target_probs": target_probs,
            "uniform_samples": uniform_samples,
            "output_tokens": output_tokens,
            "B": B,
            "T": T,
            "V": V,
        }

    def _make_accept_all_case(self, B, T, V, seed=42):
        """All draft tokens accepted: target_probs == draft_probs so alpha == 1 everywhere."""
        torch.manual_seed(seed)
        device = "cuda"

        K = min(64, V)
        draft_probs, draft_idx = self._make_sparse_probs(B, T, V, K, device)
        target_probs = draft_probs.clone()  # alpha = min(1, q/p) = 1 → always accept

        weights = draft_probs.gather(2, draft_idx)
        flat_w = weights.view(B * T, K)
        chosen = torch.multinomial(flat_w, 1).view(B, T)
        draft_tokens = draft_idx.gather(2, chosen.unsqueeze(-1)).squeeze(-1).to(torch.int32)

        # All acceptance samples set to 0 (< 1.0 = alpha) to guarantee acceptance
        uniform_samples = torch.zeros(B, T + 1, device=device)
        uniform_samples[:, T] = torch.rand(B, device=device)  # bonus sampling sample

        output_tokens = torch.zeros(B, T + 1, device=device, dtype=torch.int32)

        return {
            "draft_tokens": draft_tokens,
            "draft_probs": draft_probs,
            "target_probs": target_probs,
            "uniform_samples": uniform_samples,
            "output_tokens": output_tokens,
            "B": B,
            "T": T,
            "V": V,
        }

    def _make_reject_first_case(self, B, T, V, seed=42):
        """First draft token always rejected: draft_probs high, target low for that token."""
        torch.manual_seed(seed)
        device = "cuda"

        draft_probs = torch.softmax(torch.randn(B, T, V, device=device), dim=-1)
        target_probs = torch.softmax(torch.randn(B, T, V, device=device), dim=-1)

        flat = draft_probs.view(B * T, V)
        draft_tokens = torch.multinomial(flat, 1).view(B, T).to(torch.int32)

        # Force rejection at position 0 for every sequence:
        # set alpha[b,0] very small and uniform_sample[b,0] high enough to reject
        for b in range(B):
            tok = int(draft_tokens[b, 0].item())
            # Make draft prob ~0.9 for the chosen token (high p)
            draft_probs[b, 0] = torch.full((V,), 0.1 / max(V - 1, 1), device=device)
            draft_probs[b, 0, tok] = 0.9
            draft_probs[b, 0] = draft_probs[b, 0] / draft_probs[b, 0].sum()
            # Make target prob ~1/V for the same token (low q)
            target_probs[b, 0] = torch.ones(V, device=device) / V

        uniform_samples = torch.rand(B, T + 1, device=device)
        # Force uniform[b, 0] = 0.99 > alpha (which is ~1/V / 0.9 ≈ small)
        uniform_samples[:, 0] = 0.99

        output_tokens = torch.zeros(B, T + 1, device=device, dtype=torch.int32)

        return {
            "draft_tokens": draft_tokens,
            "draft_probs": draft_probs,
            "target_probs": target_probs,
            "uniform_samples": uniform_samples,
            "output_tokens": output_tokens,
            "B": B,
            "T": T,
            "V": V,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = "cuda"

        # B=1, T=3, V=4: position 0 accepted, position 1 rejected, token resampled
        draft_tokens = torch.tensor([[1, 2, 0]], device=device, dtype=torch.int32)

        draft_probs = torch.tensor(
            [
                [
                    [0.10, 0.60, 0.20, 0.10],  # pos 0: draft_tokens[0,0]=1, p=0.60
                    [0.10, 0.20, 0.50, 0.20],  # pos 1: draft_tokens[0,1]=2, p=0.50
                    [0.40, 0.20, 0.20, 0.20],  # pos 2: draft_tokens[0,2]=0, p=0.40
                ]
            ],
            device=device,
            dtype=torch.float32,
        )

        target_probs = torch.tensor(
            [
                [
                    [0.10, 0.50, 0.20, 0.20],  # pos 0: q=0.50, alpha=min(1,0.50/0.60)=0.833
                    [0.30, 0.20, 0.20, 0.30],  # pos 1: q=0.20, alpha=min(1,0.20/0.50)=0.400
                    [0.30, 0.20, 0.30, 0.20],  # pos 2: not reached
                ]
            ],
            device=device,
            dtype=torch.float32,
        )

        # uniform_samples[0, 0]=0.50 < 0.833 → ACCEPT token 1
        # uniform_samples[0, 1]=0.70 > 0.400 → REJECT token 2
        #   adjusted = clamp([0.20, 0, -0.30, 0.10], min=0) = [0.20, 0, 0, 0.10]
        #   normalized CDF = [0.667, 0.667, 0.667, 1.0]
        #   uniform_samples[0, T=3]=0.90 → searchsorted → token 3
        # output_tokens[0] = [1, 3, 0, 0]
        uniform_samples = torch.tensor(
            [[0.50, 0.70, 0.30, 0.90]], device=device, dtype=torch.float32
        )

        output_tokens = torch.zeros(1, 4, device=device, dtype=torch.int32)

        return {
            "draft_tokens": draft_tokens,
            "draft_probs": draft_probs,
            "target_probs": target_probs,
            "uniform_samples": uniform_samples,
            "output_tokens": output_tokens,
            "B": 1,
            "T": 3,
            "V": 4,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Edge: T=1, rejected immediately
        tests.append(self._make_reject_first_case(1, 1, 4, seed=1))

        # Edge: T=1, all accepted (bonus token sampled)
        tests.append(self._make_accept_all_case(1, 1, 4, seed=2))

        # Edge: T=2, first rejected
        tests.append(self._make_reject_first_case(1, 2, 8, seed=3))

        # Edge: T=4, all accepted
        tests.append(self._make_accept_all_case(2, 4, 8, seed=4))

        # Zero uniform_samples acceptance values → force rejection at pos 0 (unless alpha=1)
        tests.append(self._make_reject_first_case(4, 4, 16, seed=5))

        # Power-of-2 vocab, mixed acceptance
        tests.append(self._make_test_case(4, 8, 64, seed=10))

        # Larger vocab, mixed acceptance
        tests.append(self._make_test_case(8, 8, 256, seed=20))

        # Non-power-of-2 vocab
        tests.append(self._make_test_case(4, 6, 30, seed=30))

        # All sequences accept all tokens (bonus sampling)
        tests.append(self._make_accept_all_case(8, 8, 128, seed=40))

        # Realistic small batch
        tests.append(self._make_test_case(16, 8, 1000, seed=50))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        # B=64 sequences, T=8 draft tokens, V=32768 (Mistral/LLaMA-2 vocab size)
        return self._make_test_case(64, 8, 32768, seed=0)
