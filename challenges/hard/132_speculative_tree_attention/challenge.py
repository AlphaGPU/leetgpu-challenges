import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Speculative Tree Attention"
    # Softmax-weighted averages of O(1) values over up to ~8K keys; float32
    # accumulation-order differences stay near 1e-6, while attending to a sibling
    # instead of an ancestor, dropping the prefix or skipping the 1/sqrt(D) scale
    # changes outputs by >> 1e-4.
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        K_tree: torch.Tensor,
        V_tree: torch.Tensor,
        parents: torch.Tensor,
        output: torch.Tensor,
        P: int,
        T: int,
        H: int,
        D: int,
    ):
        assert Q.shape == (T, H, D)
        assert K_cache.shape == (P, H, D)
        assert V_cache.shape == (P, H, D)
        assert K_tree.shape == (T, H, D)
        assert V_tree.shape == (T, H, D)
        assert parents.shape == (T,)
        assert output.shape == (T, H, D)
        assert (
            Q.dtype
            == K_cache.dtype
            == V_cache.dtype
            == K_tree.dtype
            == V_tree.dtype
            == output.dtype
            == torch.float32
        )
        assert parents.dtype == torch.int32

        # Ancestor-or-self mask over tree nodes; parents[i] < i so one forward pass suffices.
        par = parents.cpu().tolist()
        allowed = torch.zeros(T, T, dtype=torch.bool)
        for i in range(T):
            if par[i] >= 0:
                allowed[i] = allowed[par[i]]
            allowed[i, i] = True
        allowed = allowed.to(Q.device)

        K = torch.cat([K_cache, K_tree], dim=0)  # (P + T, H, D)
        V = torch.cat([V_cache, V_tree], dim=0)
        mask = torch.cat(
            [torch.ones(T, P, dtype=torch.bool, device=Q.device), allowed], dim=1
        )  # (T, P + T)

        scores = torch.einsum("thd,shd->hts", Q, K) / math.sqrt(D)
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        output.copy_(torch.einsum("hts,shd->thd", attn, V))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K_cache": (ctypes.POINTER(ctypes.c_float), "in"),
            "V_cache": (ctypes.POINTER(ctypes.c_float), "in"),
            "K_tree": (ctypes.POINTER(ctypes.c_float), "in"),
            "V_tree": (ctypes.POINTER(ctypes.c_float), "in"),
            "parents": (ctypes.POINTER(ctypes.c_int), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "P": (ctypes.c_int, "in"),
            "T": (ctypes.c_int, "in"),
            "H": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
        }

    def _make(self, P, T, H, D, parents, seed=0, scale=1.0):
        torch.manual_seed(seed)

        def rnd(*shape):
            return torch.empty(*shape, device=self.device, dtype=torch.float32).uniform_(
                -scale, scale
            )

        return {
            "Q": rnd(T, H, D),
            "K_cache": rnd(P, H, D),
            "V_cache": rnd(P, H, D),
            "K_tree": rnd(T, H, D),
            "V_tree": rnd(T, H, D),
            "parents": torch.tensor(parents, device=self.device, dtype=torch.int32),
            "output": torch.zeros(T, H, D, device=self.device, dtype=torch.float32),
            "P": P,
            "T": T,
            "H": H,
            "D": D,
        }

    @staticmethod
    def _random_tree(T, seed):
        g = torch.Generator().manual_seed(seed)
        par = [-1]
        for i in range(1, T):
            par.append(int(torch.randint(-1, i, (1,), generator=g).item()))
        return par

    @staticmethod
    def _chain(T):
        return [i - 1 for i in range(T)]

    @staticmethod
    def _wide(T, branching):
        # Breadth-first complete tree: node i's parent is (i - 1) // branching, roots at -1.
        return [-1 if i == 0 else (i - 1) // branching for i in range(T)]

    def generate_example_test(self) -> Dict[str, Any]:
        dev, f32 = self.device, torch.float32
        return {
            "Q": torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]], device=dev, dtype=f32),
            "K_cache": torch.tensor([[[1.0, 0.0]]], device=dev, dtype=f32),
            "V_cache": torch.tensor([[[1.0, 2.0]]], device=dev, dtype=f32),
            "K_tree": torch.tensor(
                [[[0.0, 1.0]], [[1.0, 1.0]], [[0.0, 0.0]]], device=dev, dtype=f32
            ),
            "V_tree": torch.tensor(
                [[[3.0, 0.0]], [[0.0, 3.0]], [[2.0, 2.0]]], device=dev, dtype=f32
            ),
            "parents": torch.tensor([-1, 0, 0], device=dev, dtype=torch.int32),
            "output": torch.zeros(3, 1, 2, device=dev, dtype=f32),
            "P": 1,
            "T": 3,
            "H": 1,
            "D": 2,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = [self.generate_example_test()]

        # Single draft token with a single prefix token: degenerates to plain decode.
        tests.append(self._make(1, 1, 1, 1, [-1], seed=1))

        # Pure chain: equivalent to causal attention over the draft; catches a reversed mask.
        tests.append(self._make(5, 8, 2, 4, self._chain(8), seed=2))

        # Several independent roots (forest): siblings must never see each other.
        tests.append(self._make(3, 6, 1, 8, [-1, -1, -1, 0, 1, 2], seed=3))

        # Zero queries: uniform attention over prefix + ancestors.
        z = self._make(4, 4, 2, 4, [-1, 0, 1, 1], seed=4)
        z["Q"].zero_()
        tests.append(z)

        # Negative-only values.
        neg = self._make(16, 7, 2, 16, self._random_tree(7, 5), seed=5)
        neg["V_cache"].abs_().neg_()
        neg["V_tree"].abs_().neg_()
        tests.append(neg)

        # Large scores: softmax without max-subtraction overflows.
        tests.append(self._make(33, 10, 2, 32, self._random_tree(10, 6), seed=6, scale=6.0))

        # Wide binary tree, non-power-of-two prefix and head dim.
        tests.append(self._make(100, 15, 3, 24, self._wide(15, 2), seed=7))

        # Power-of-two shapes with a random deep tree.
        tests.append(self._make(256, 32, 4, 64, self._random_tree(32, 8), seed=8))

        # Realistic EAGLE/Medusa-style tree over a long prefix.
        tests.append(self._make(2048, 64, 8, 128, self._wide(64, 4), seed=9))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._make(8192, 64, 32, 128, self._random_tree(64, 0), seed=0)
