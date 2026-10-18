import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Paged Chunked Prefill Attention"
    # Outputs are softmax-weighted averages of O(1) values. The float64 reference keeps
    # float32 / online-softmax / split-K solutions within ~1e-5. Attending one key past the
    # causal limit, using the chunk-local query index as the position, reading the wrong
    # KV head, or ignoring the block table is off by far more than 1e-3 on the small
    # targeted cases below.
    atol = 1e-03
    rtol = 1e-03
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        output: torch.Tensor,
        num_seqs: int,
        total_q: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        max_blocks_per_seq: int,
    ):
        assert Q.shape == output.shape == (total_q, num_q_heads, head_dim)
        assert K_cache.shape[1:] == (block_size, num_kv_heads, head_dim)
        assert V_cache.shape == K_cache.shape
        assert block_table.shape == (num_seqs, max_blocks_per_seq)
        assert context_lens.shape == (num_seqs,)
        assert cu_seqlens_q.shape == (num_seqs + 1,)
        assert Q.dtype == K_cache.dtype == V_cache.dtype == output.dtype == torch.float32
        assert block_table.dtype == context_lens.dtype == cu_seqlens_q.dtype == torch.int32
        assert num_q_heads % num_kv_heads == 0

        group = num_q_heads // num_kv_heads
        scale = 1.0 / math.sqrt(head_dim)
        ctx_list = context_lens.tolist()
        cu_list = cu_seqlens_q.tolist()

        # One sequence at a time keeps memory at (num_q_heads, q_len, ctx_len) instead of a
        # padded (num_seqs, num_q_heads, max_q, max_ctx) score tensor.
        for s in range(num_seqs):
            q0, q1, ctx = cu_list[s], cu_list[s + 1], ctx_list[s]
            q_len = q1 - q0
            n_blk = (ctx + block_size - 1) // block_size
            blocks = block_table[s, :n_blk].long()

            # Gather this sequence's logical KV [0, ctx) from its scattered physical blocks.
            k = K_cache[blocks].reshape(n_blk * block_size, num_kv_heads, head_dim)[:ctx]
            v = V_cache[blocks].reshape(n_blk * block_size, num_kv_heads, head_dim)[:ctx]
            # GQA: query head h reads KV head h // group.
            k = k.double().repeat_interleave(group, dim=1)
            v = v.double().repeat_interleave(group, dim=1)
            q = Q[q0:q1].double()

            scores = torch.einsum("qhd,khd->hqk", q, k) * scale
            # The chunk is the last q_len tokens of the sequence: query i sits at absolute
            # position ctx - q_len + i and may attend to keys 0..that position inclusive.
            q_pos = torch.arange(q_len, device=Q.device) + (ctx - q_len)
            k_pos = torch.arange(ctx, device=Q.device)
            allowed = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
            scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            output[q0:q1] = torch.einsum("hqk,khd->qhd", attn, v).float()

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K_cache": (ctypes.POINTER(ctypes.c_float), "in"),
            "V_cache": (ctypes.POINTER(ctypes.c_float), "in"),
            "block_table": (ctypes.POINTER(ctypes.c_int), "in"),
            "context_lens": (ctypes.POINTER(ctypes.c_int), "in"),
            "cu_seqlens_q": (ctypes.POINTER(ctypes.c_int), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "num_seqs": (ctypes.c_int, "in"),
            "total_q": (ctypes.c_int, "in"),
            "num_q_heads": (ctypes.c_int, "in"),
            "num_kv_heads": (ctypes.c_int, "in"),
            "head_dim": (ctypes.c_int, "in"),
            "block_size": (ctypes.c_int, "in"),
            "max_blocks_per_seq": (ctypes.c_int, "in"),
        }

    def _make(
        self,
        q_lens,
        ctx_lens,
        num_q_heads,
        num_kv_heads,
        head_dim,
        block_size,
        seed=0,
        q_scale=1.0,
        zero_q=False,
    ):
        assert len(q_lens) == len(ctx_lens)
        assert all(1 <= q <= c for q, c in zip(q_lens, ctx_lens))
        torch.manual_seed(seed)
        dev, f32 = self.device, torch.float32
        num_seqs = len(q_lens)
        total_q = sum(q_lens)

        blocks_needed = [(c + block_size - 1) // block_size for c in ctx_lens]
        max_blocks_per_seq = max(blocks_needed)
        used = sum(blocks_needed)
        # Over-allocate the pool and scatter every sequence's blocks through it, so a solution
        # that ignores block_table and reads the cache contiguously fails.
        pool = used + max(2, used // 2)
        perm = torch.randperm(pool).tolist()
        spare = perm[used:]
        table, nxt = [], 0
        for s in range(num_seqs):
            row = perm[nxt : nxt + blocks_needed[s]]
            nxt += blocks_needed[s]
            # Padding entries point at unused blocks: reading them gives wrong answers.
            row += [spare[i % len(spare)] for i in range(max_blocks_per_seq - len(row))]
            table.append(row)

        kv_shape = (pool, block_size, num_kv_heads, head_dim)
        K_cache = torch.randn(kv_shape, device=dev, dtype=f32)
        V_cache = torch.randn(kv_shape, device=dev, dtype=f32)
        if zero_q:
            Q = torch.zeros(total_q, num_q_heads, head_dim, device=dev, dtype=f32)
        else:
            Q = torch.randn(total_q, num_q_heads, head_dim, device=dev, dtype=f32) * q_scale

        cu = [0]
        for q in q_lens:
            cu.append(cu[-1] + q)
        return {
            "Q": Q,
            "K_cache": K_cache,
            "V_cache": V_cache,
            "block_table": torch.tensor(table, device=dev, dtype=torch.int32),
            "context_lens": torch.tensor(ctx_lens, device=dev, dtype=torch.int32),
            "cu_seqlens_q": torch.tensor(cu, device=dev, dtype=torch.int32),
            "output": torch.full((total_q, num_q_heads, head_dim), 123.0, device=dev, dtype=f32),
            "num_seqs": num_seqs,
            "total_q": total_q,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "block_size": block_size,
            "max_blocks_per_seq": max_blocks_per_seq,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dev, f32 = self.device, torch.float32
        # 1 sequence, context 3, chunk of the last 2 tokens (positions 1 and 2).
        # 2 query heads share 1 KV head. block_size 2; logical block 0 -> physical 2,
        # logical block 1 -> physical 0. Physical block 1 and slot 1 of block 0 are unused.
        K_cache = torch.tensor(
            [
                [[[0.0, 1.0]], [[9.0, 9.0]]],
                [[[9.0, 9.0]], [[9.0, 9.0]]],
                [[[1.0, 0.0]], [[1.0, 0.0]]],
            ],
            device=dev,
            dtype=f32,
        )
        V_cache = torch.tensor(
            [
                [[[3.0, 3.0]], [[9.0, 9.0]]],
                [[[9.0, 9.0]], [[9.0, 9.0]]],
                [[[3.0, 0.0]], [[0.0, 3.0]]],
            ],
            device=dev,
            dtype=f32,
        )
        Q = torch.tensor(
            [[[0.0, 0.0], [2.0, 0.0]], [[0.0, 0.0], [2.0, 0.0]]], device=dev, dtype=f32
        )
        return {
            "Q": Q,
            "K_cache": K_cache,
            "V_cache": V_cache,
            "block_table": torch.tensor([[2, 0]], device=dev, dtype=torch.int32),
            "context_lens": torch.tensor([3], device=dev, dtype=torch.int32),
            "cu_seqlens_q": torch.tensor([0, 2], device=dev, dtype=torch.int32),
            "output": torch.full((2, 2, 2), 123.0, device=dev, dtype=f32),
            "num_seqs": 1,
            "total_q": 2,
            "num_q_heads": 2,
            "num_kv_heads": 1,
            "head_dim": 2,
            "block_size": 2,
            "max_blocks_per_seq": 2,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Hand-computed example (values checked independently in challenge.html): token 0
        # must not see key 2, so head 0 gives [1.5, 1.5], not [2, 2].
        tests.append(self.generate_example_test())

        # Smallest case: one decode token over a one-token context, block_size 1.
        tests.append(self._make([1], [1], 1, 1, 4, 1, seed=1))

        # Fresh prompt (q_len == ctx): plain causal prefill; catches a missing causal mask.
        tests.append(self._make([3], [3], 2, 2, 8, 4, seed=2))

        # Pure decode batch with GQA group 4 and partial trailing blocks: catches mapping
        # query head h to KV head h % num_kv_heads instead of h // group.
        tests.append(self._make([1, 1, 1, 1], [1, 5, 17, 33], 8, 2, 16, 4, seed=3))

        # Chunk after a long prefix (q_len << ctx): catches using the chunk-local index i
        # instead of ctx - q_len + i as the query position (would hide most of the prefix).
        tests.append(self._make([4, 7], [64, 100], 4, 2, 32, 16, seed=4))

        # Zero queries: each output is the plain mean of V over its causal prefix, so any
        # off-by-one in the mask boundary or a read past ctx (padding slots) shows directly.
        tests.append(self._make([5, 3], [5, 30], 4, 1, 8, 8, seed=5, zero_q=True))

        # Large-magnitude scores (|score| up to ~100): exp overflows in float32 unless the
        # row max is subtracted; MQA (one KV head shared by 8 query heads).
        tests.append(self._make([16, 1], [48, 255], 8, 1, 64, 16, seed=6, q_scale=20.0))

        # MHA (num_q_heads == num_kv_heads), non-power-of-two lengths, block_size 64.
        tests.append(self._make([30, 1, 100], [30, 1000, 255], 4, 4, 64, 64, seed=7))

        # Mixed vLLM-style batch: decodes, a fresh prompt and continued chunks, with sizes
        # that straddle block and tile boundaries.
        tests.append(
            self._make(
                [1, 128, 1, 63, 256, 1], [2048, 128, 511, 777, 1300, 4096], 16, 4, 128, 16, seed=8
            )
        )

        # Realistic Llama-style heads (32 query / 8 KV, head_dim 128) with 512-token chunks.
        tests.append(self._make([512, 512, 1, 1], [512, 2000, 3000, 8192], 32, 8, 128, 16, seed=9))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # One continuous-batching step: 4 prefill chunks of 512 tokens (one fresh prompt,
        # three continuing prompts) plus 28 decode sequences, Llama-3-8B attention heads
        # (32 query / 8 KV, head_dim 128), vLLM block_size 16.
        q_lens = [512] * 4 + [1] * 28
        ctx_lens = [512, 1536, 2560, 4096] + [1024 + 128 * i for i in range(28)]
        return self._make(q_lens, ctx_lens, 32, 8, 128, 16, seed=0)
