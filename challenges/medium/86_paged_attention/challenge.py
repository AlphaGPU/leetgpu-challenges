import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Paged KV-Cache Attention",
            atol=1e-04,
            rtol=1e-04,
            num_gpus=1,
            access_tier="free",
        )

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
        assert Q.device.type == "cuda"
        assert K_cache.device.type == "cuda"
        assert V_cache.device.type == "cuda"
        assert block_table.device.type == "cuda"
        assert context_lens.device.type == "cuda"
        assert output.device.type == "cuda"

        scale = 1.0 / math.sqrt(head_dim)

        for s in range(batch_size):
            ctx_len = context_lens[s].item()
            n_blocks = (ctx_len + block_size - 1) // block_size

            # Gather the physical blocks assigned to this sequence
            phys_blocks = block_table[s, :n_blocks].long()  # (n_blocks,)

            # Gather K and V: (n_blocks, block_size, num_heads, head_dim)
            K_blocks = K_cache[phys_blocks]
            V_blocks = V_cache[phys_blocks]

            # Flatten to (n_blocks * block_size, num_heads, head_dim) and trim
            K_seq = K_blocks.reshape(-1, num_heads, head_dim)[
                :ctx_len
            ]  # (ctx_len, num_heads, head_dim)
            V_seq = V_blocks.reshape(-1, num_heads, head_dim)[:ctx_len]

            # Transpose to (num_heads, ctx_len, head_dim)
            K_seq = K_seq.transpose(0, 1).contiguous()
            V_seq = V_seq.transpose(0, 1).contiguous()

            # Q[s]: (num_heads, head_dim) -> (num_heads, 1, head_dim)
            q = Q[s].unsqueeze(1)

            # Scaled dot-product: (num_heads, 1, ctx_len)
            scores = torch.bmm(q, K_seq.transpose(1, 2)) * scale
            attn_weights = torch.softmax(scores, dim=-1)

            # Weighted sum: (num_heads, 1, head_dim) -> (num_heads, head_dim)
            out = torch.bmm(attn_weights, V_seq).squeeze(1)
            output[s].copy_(out)

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

        max_ctx = max(context_lens)
        max_blocks_per_seq = (max_ctx + block_size - 1) // block_size

        # Allocate exactly the blocks needed, assigned sequentially
        total_blocks = sum((cl + block_size - 1) // block_size for cl in context_lens)

        device = "cuda"
        dtype = torch.float32

        if zero_q:
            Q = torch.zeros(batch_size, num_heads, head_dim, device=device, dtype=dtype)
        else:
            Q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=dtype)

        K_cache = torch.randn(
            total_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype
        )
        V_cache = torch.randn(
            total_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype
        )

        block_table = torch.zeros(batch_size, max_blocks_per_seq, device=device, dtype=torch.int32)
        ctx_lens_tensor = torch.tensor(context_lens, device=device, dtype=torch.int32)

        # Assign physical blocks sequentially per sequence
        block_idx = 0
        for s in range(batch_size):
            n_blocks = (context_lens[s] + block_size - 1) // block_size
            for b in range(n_blocks):
                block_table[s, b] = block_idx
                block_idx += 1

        output = torch.zeros(batch_size, num_heads, head_dim, device=device, dtype=dtype)

        return {
            "Q": Q,
            "K_cache": K_cache,
            "V_cache": V_cache,
            "block_table": block_table,
            "context_lens": ctx_lens_tensor,
            "output": output,
            "batch_size": batch_size,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "block_size": block_size,
            "max_blocks_per_seq": max_blocks_per_seq,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = "cuda"
        dtype = torch.float32

        # batch=1, heads=1, head_dim=4, block_size=2, ctx_len=2
        # Q · K / sqrt(4): [1,1,0,0]·[1,0,0,0]/2 = 0.5, [1,1,0,0]·[0,1,0,0]/2 = 0.5
        # attn = softmax([0.5, 0.5]) = [0.5, 0.5]
        # output = 0.5*[2,0,0,0] + 0.5*[0,4,0,0] = [1, 2, 0, 0]
        Q = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]], device=device, dtype=dtype)  # (1, 1, 4)
        K_cache = torch.tensor(
            [[[[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]]]],
            device=device,
            dtype=dtype,
        )  # (1 block, block_size=2, 1 head, head_dim=4)
        V_cache = torch.tensor(
            [[[[2.0, 0.0, 0.0, 0.0]], [[0.0, 4.0, 0.0, 0.0]]]],
            device=device,
            dtype=dtype,
        )
        block_table = torch.tensor(
            [[0]], device=device, dtype=torch.int32
        )  # seq 0 -> physical block 0
        context_lens = torch.tensor([2], device=device, dtype=torch.int32)
        output = torch.zeros(1, 1, 4, device=device, dtype=dtype)

        return {
            "Q": Q,
            "K_cache": K_cache,
            "V_cache": V_cache,
            "block_table": block_table,
            "context_lens": context_lens,
            "output": output,
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

        # Edge case: ctx_len equals block_size exactly
        tests.append(self._make_test_case(1, 2, 8, 4, 4))

        # Zero query: softmax is uniform, output is mean of V
        tests.append(self._make_test_case(2, 2, 8, 4, 8, zero_q=True))

        # Variable context lengths within a batch
        tests.append(self._make_test_case(4, 4, 32, 16, [16, 32, 48, 64]))

        # Power-of-2 context lengths
        tests.append(self._make_test_case(4, 4, 32, 16, 32))

        # Power-of-2, larger
        tests.append(self._make_test_case(4, 8, 64, 16, 128))

        # Non-power-of-2 context length
        tests.append(self._make_test_case(2, 4, 32, 16, 30))

        # Non-power-of-2, straddles multiple blocks
        tests.append(self._make_test_case(4, 4, 64, 16, 100))

        # Mixed variable lengths with non-power-of-2
        tests.append(self._make_test_case(4, 8, 64, 16, [50, 100, 150, 200]))

        # Realistic: LLaMA-3 8B style (8 Q heads), shorter context
        tests.append(self._make_test_case(4, 8, 128, 16, 256))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        # Realistic LLM decode: batch=8, 32 heads, head_dim=128, block_size=16, ctx_len=2048
        return self._make_test_case(8, 32, 128, 16, 2048)
