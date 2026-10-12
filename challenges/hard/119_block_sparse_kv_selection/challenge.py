import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Block-Sparse KV Selection Attention"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        output: torch.Tensor,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        block_size: int,
        num_selected: int,
    ):
        assert Q.shape == (num_heads, head_dim)
        assert K.shape == (num_heads, seq_len, head_dim)
        assert V.shape == (num_heads, seq_len, head_dim)
        assert output.shape == (num_heads, head_dim)
        assert Q.dtype == torch.float32
        assert K.dtype == torch.float32
        assert V.dtype == torch.float32
        assert output.dtype == torch.float32

        num_blocks = (seq_len + block_size - 1) // block_size
        pad = num_blocks * block_size - seq_len

        # Compressed cache: one mean-pooled key per block. The tail block averages
        # only the positions that actually exist.
        K_padded = torch.nn.functional.pad(K, (0, 0, 0, pad))
        K_blocks = K_padded.view(num_heads, num_blocks, block_size, head_dim)
        position = torch.arange(num_blocks * block_size, device=K.device)
        valid = (position < seq_len).view(num_blocks, block_size)
        counts = valid.sum(dim=1).to(K.dtype)
        block_mean = K_blocks.sum(dim=2) / counts.view(1, num_blocks, 1)

        # Block importance score: raw dot product of the query with the pooled key.
        block_scores = torch.bmm(Q.unsqueeze(1), block_mean.transpose(1, 2)).squeeze(1)

        # Rank each block: blocks with a strictly higher score come first, and equal
        # scores are ordered by block index so the selection is deterministic.
        index = torch.arange(num_blocks, device=K.device)
        higher = block_scores.unsqueeze(1) > block_scores.unsqueeze(2)
        tied_earlier = (block_scores.unsqueeze(1) == block_scores.unsqueeze(2)) & (
            index.view(1, 1, num_blocks) < index.view(1, num_blocks, 1)
        )
        rank = (higher | tied_earlier).sum(dim=2)
        selected = rank < min(num_selected, num_blocks)

        # Expand the block mask to a position mask and run a masked softmax attention.
        position_mask = selected.unsqueeze(2).expand(num_heads, num_blocks, block_size)
        position_mask = position_mask.reshape(num_heads, num_blocks * block_size)[:, :seq_len]

        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.bmm(Q.unsqueeze(1), K.transpose(1, 2)).squeeze(1) * scale
        scores = scores.masked_fill(~position_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        output.copy_(torch.bmm(weights.unsqueeze(1), V).squeeze(1))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "num_heads": (ctypes.c_int, "in"),
            "seq_len": (ctypes.c_int, "in"),
            "head_dim": (ctypes.c_int, "in"),
            "block_size": (ctypes.c_int, "in"),
            "num_selected": (ctypes.c_int, "in"),
        }

    def _make_test_case(
        self,
        num_heads,
        seq_len,
        head_dim,
        block_size,
        num_selected,
        zero_q=False,
        zero_k=False,
        seed=None,
    ):
        dtype = torch.float32
        device = self.device
        if seed is not None:
            torch.manual_seed(seed)

        if zero_q:
            Q = torch.zeros(num_heads, head_dim, device=device, dtype=dtype)
        else:
            Q = torch.randn(num_heads, head_dim, device=device, dtype=dtype)

        if zero_k:
            K = torch.zeros(num_heads, seq_len, head_dim, device=device, dtype=dtype)
        else:
            K = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=dtype)

        V = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=dtype)
        output = torch.empty(num_heads, head_dim, device=device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "output": output,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "block_size": block_size,
            "num_selected": num_selected,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = self.device
        num_heads, seq_len, head_dim, block_size, num_selected = 2, 4, 2, 2, 1

        Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device, dtype=dtype)
        K = torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            ],
            device=device,
            dtype=dtype,
        )
        V = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            ],
            device=device,
            dtype=dtype,
        )
        output = torch.empty(num_heads, head_dim, device=device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "output": output,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "block_size": block_size,
            "num_selected": num_selected,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Edge case: a single cached position forming a single block
        tests.append(self._make_test_case(1, 1, 8, 1, 1, seed=0))

        # Edge case: partial tail block (blocks are [0, 1] and [2])
        tests.append(self._make_test_case(2, 3, 8, 2, 1, seed=2))

        # Edge case: num_selected exceeds the block count, so every block is kept
        tests.append(self._make_test_case(2, 10, 8, 4, 5, seed=3))

        # Zero query: every block score ties, so the lowest block indices win
        tests.append(self._make_test_case(1, 16, 16, 4, 2, zero_q=True, seed=4))

        # Zero keys: uniform attention over the first num_selected blocks
        tests.append(self._make_test_case(4, 64, 32, 16, 2, zero_k=True, seed=5))

        # Power-of-2 shapes
        tests.append(self._make_test_case(8, 128, 64, 16, 4, seed=6))
        tests.append(self._make_test_case(4, 512, 128, 32, 8, seed=7))

        # Non-power-of-2 shapes with partial tail blocks
        tests.append(self._make_test_case(2, 30, 64, 8, 3, seed=8))
        tests.append(self._make_test_case(6, 255, 100, 25, 4, seed=9))

        # Realistic decode step: 1K context, 64-position blocks, 8 blocks kept
        tests.append(self._make_test_case(16, 1024, 128, 64, 8, seed=10))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # Sparse decode step: 32 heads, 8K context, 128 blocks of 64, only 32 kept
        return self._make_test_case(32, 8192, 128, 64, 32, seed=42)
