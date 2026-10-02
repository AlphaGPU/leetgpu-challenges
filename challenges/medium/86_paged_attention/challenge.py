import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Paged KV-Cache Attention"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
        output: torch.Tensor,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        block_size: int,
        max_blocks_per_seq: int,
    ):
        assert Q.shape == (batch_size, num_heads, head_dim)
        assert K_cache.shape[1] == block_size
        assert K_cache.shape[2] == num_heads
        assert K_cache.shape[3] == head_dim
        assert V_cache.shape == K_cache.shape
        assert block_table.shape == (batch_size, max_blocks_per_seq)
        assert context_lens.shape == (batch_size,)
        assert output.shape == (batch_size, num_heads, head_dim)
        assert Q.dtype == K_cache.dtype == V_cache.dtype == output.dtype == torch.float32
        assert block_table.dtype == context_lens.dtype == torch.int32

        scale = 1.0 / math.sqrt(head_dim)

        # Logical token positions covered by the block table, padded up to a whole
        # number of blocks so the gather shape is static.
        max_ctx = max_blocks_per_seq * block_size
        pos = torch.arange(max_ctx, device=Q.device)
        logical_block = torch.div(pos, block_size, rounding_mode="floor")
        offset = pos - logical_block * block_size

        # Map each logical position to (physical block, offset within block)
        phys_block = block_table.long()[:, logical_block]  # (batch_size, max_ctx)
        within_block = offset.unsqueeze(0).expand(batch_size, max_ctx)

        # (batch_size, max_ctx, num_heads, head_dim)
        K_gathered = K_cache[phys_block, within_block]
        V_gathered = V_cache[phys_block, within_block]

        scores = torch.einsum("bhd,bthd->bth", Q, K_gathered) * scale
        valid = pos.unsqueeze(0) < context_lens.long().unsqueeze(1)  # (batch_size, max_ctx)
        scores = scores.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        attn_weights = torch.softmax(scores, dim=1)

        output.copy_(torch.einsum("bth,bthd->bhd", attn_weights, V_gathered))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K_cache": (ctypes.POINTER(ctypes.c_float), "in"),
            "V_cache": (ctypes.POINTER(ctypes.c_float), "in"),
            "block_table": (ctypes.POINTER(ctypes.c_int), "in"),
            "context_lens": (ctypes.POINTER(ctypes.c_int), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "batch_size": (ctypes.c_int, "in"),
            "num_heads": (ctypes.c_int, "in"),
            "head_dim": (ctypes.c_int, "in"),
            "block_size": (ctypes.c_int, "in"),
            "max_blocks_per_seq": (ctypes.c_int, "in"),
        }

    def _make_test_case(
        self, batch_size, num_heads, head_dim, block_size, context_lens, zero_q=False
    ):
        if isinstance(context_lens, int):
            context_lens = [context_lens] * batch_size

        blocks_needed = [(cl + block_size - 1) // block_size for cl in context_lens]
        max_blocks_per_seq = max(blocks_needed)
        total_needed = sum(blocks_needed)

        # Over-allocate the pool and scatter each sequence's blocks through it, so a
        # solution that ignores block_table and reads the cache contiguously fails.
        pool_blocks = total_needed + max(1, total_needed // 2)
        perm = torch.randperm(pool_blocks).tolist()

        table = [[0] * max_blocks_per_seq for _ in range(batch_size)]
        nxt = 0
        for s in range(batch_size):
            for b in range(blocks_needed[s]):
                table[s][b] = perm[nxt]
                nxt += 1

        dtype = torch.float32
        if zero_q:
            Q = torch.zeros(batch_size, num_heads, head_dim, device=self.device, dtype=dtype)
        else:
            Q = torch.randn(batch_size, num_heads, head_dim, device=self.device, dtype=dtype)

        K_cache = torch.randn(
            pool_blocks, block_size, num_heads, head_dim, device=self.device, dtype=dtype
        )
        V_cache = torch.randn(
            pool_blocks, block_size, num_heads, head_dim, device=self.device, dtype=dtype
        )

        return {
            "Q": Q,
            "K_cache": K_cache,
            "V_cache": V_cache,
            "block_table": torch.tensor(table, device=self.device, dtype=torch.int32),
            "context_lens": torch.tensor(context_lens, device=self.device, dtype=torch.int32),
            "output": torch.zeros(batch_size, num_heads, head_dim, device=self.device, dtype=dtype),
            "batch_size": batch_size,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "block_size": block_size,
            "max_blocks_per_seq": max_blocks_per_seq,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32

        # batch=1, heads=1, head_dim=4, block_size=2, ctx_len=2, seq 0 -> physical block 1
        # scores = Q . K / sqrt(4) = [0.5, 0.5]; softmax([0.5, 0.5]) = [0.5, 0.5]
        # output = 0.5*[2,0,0,0] + 0.5*[0,4,0,0] = [1, 2, 0, 0]
        Q = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]], device=self.device, dtype=dtype)
        K_cache = torch.tensor(
            [
                [[[9.0, 9.0, 9.0, 9.0]], [[9.0, 9.0, 9.0, 9.0]]],
                [[[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]]],
            ],
            device=self.device,
            dtype=dtype,
        )
        V_cache = torch.tensor(
            [
                [[[9.0, 9.0, 9.0, 9.0]], [[9.0, 9.0, 9.0, 9.0]]],
                [[[2.0, 0.0, 0.0, 0.0]], [[0.0, 4.0, 0.0, 0.0]]],
            ],
            device=self.device,
            dtype=dtype,
        )

        return {
            "Q": Q,
            "K_cache": K_cache,
            "V_cache": V_cache,
            "block_table": torch.tensor([[1]], device=self.device, dtype=torch.int32),
            "context_lens": torch.tensor([2], device=self.device, dtype=torch.int32),
            "output": torch.zeros(1, 1, 4, device=self.device, dtype=dtype),
            "batch_size": 1,
            "num_heads": 1,
            "head_dim": 4,
            "block_size": 2,
            "max_blocks_per_seq": 1,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Edge case: single KV token
        tests.append(self._make_test_case(1, 1, 4, 2, 1))

        # Edge case: context length equals block size exactly
        tests.append(self._make_test_case(1, 2, 8, 4, 4))

        # Zero query: softmax is uniform, output is the mean of V
        tests.append(self._make_test_case(2, 2, 8, 4, 8, zero_q=True))

        # Edge case: partially filled trailing block (ctx_len = 3, block_size = 4)
        tests.append(self._make_test_case(3, 2, 8, 4, [1, 2, 3]))

        # Variable context lengths within a batch
        tests.append(self._make_test_case(4, 4, 32, 16, [16, 32, 48, 64]))

        # Power-of-2 context lengths
        tests.append(self._make_test_case(4, 8, 64, 16, 128))

        # Non-power-of-2 context length
        tests.append(self._make_test_case(2, 4, 32, 16, 30))

        # Non-power-of-2, straddles multiple blocks
        tests.append(self._make_test_case(4, 4, 64, 16, 100))

        # Mixed variable lengths with non-power-of-2
        tests.append(self._make_test_case(4, 8, 64, 16, [50, 100, 150, 200]))

        # Realistic: 8 query heads, longer context
        tests.append(self._make_test_case(4, 8, 128, 16, 256))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        # Realistic LLM decode: batch=8, 32 heads, head_dim=128, block_size=16, ctx_len=2048
        return self._make_test_case(8, 32, 128, 16, 2048)
