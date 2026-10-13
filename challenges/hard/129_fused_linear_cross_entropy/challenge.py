import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Fused Linear Cross Entropy"
    # The loss is a mean of per-row log-sum-exp values over large vocabularies;
    # float32 accumulation-order differences across a K=1,024 dot product and a
    # V=32,768 reduction stay well below 1e-3, while a wrong target index, a
    # missing max-subtraction (inf/nan) or a sum instead of a mean is off by >> 1e-3.
    atol = 1e-03
    rtol = 1e-03
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
        N: int,
        D: int,
        V: int,
    ):
        assert X.shape == (N, D)
        assert W.shape == (V, D)
        assert targets.shape == (N,)
        assert loss.shape == (1,)
        assert X.dtype == W.dtype == loss.dtype == torch.float32
        assert targets.dtype == torch.int32

        logits = X @ W.T
        lse = torch.logsumexp(logits, dim=1)
        target_logit = torch.gather(logits, 1, targets.long().unsqueeze(1)).squeeze(1)
        loss.copy_((lse - target_logit).mean().reshape(1))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "X": (ctypes.POINTER(ctypes.c_float), "in"),
            "W": (ctypes.POINTER(ctypes.c_float), "in"),
            "targets": (ctypes.POINTER(ctypes.c_int), "in"),
            "loss": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
            "V": (ctypes.c_int, "in"),
        }

    def _make(self, N, D, V, scale=1.0, seed=0, targets=None):
        torch.manual_seed(seed)
        X = torch.empty(N, D, device=self.device, dtype=torch.float32).uniform_(-1.0, 1.0)
        W = torch.empty(V, D, device=self.device, dtype=torch.float32).uniform_(-1.0, 1.0)
        W.mul_(scale / (D**0.5))
        if targets is None:
            t = torch.randint(0, V, (N,), device=self.device, dtype=torch.int32)
        else:
            t = torch.tensor(targets, device=self.device, dtype=torch.int32)
        return {
            "X": X,
            "W": W,
            "targets": t,
            "loss": torch.zeros(1, device=self.device, dtype=torch.float32),
            "N": N,
            "D": D,
            "V": V,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        X = torch.tensor([[1.0, 0.0], [0.0, 2.0]], device=self.device, dtype=torch.float32)
        W = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], device=self.device, dtype=torch.float32
        )
        targets = torch.tensor([0, 2], device=self.device, dtype=torch.int32)
        return {
            "X": X,
            "W": W,
            "targets": targets,
            "loss": torch.zeros(1, device=self.device, dtype=torch.float32),
            "N": 2,
            "D": 2,
            "V": 3,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = [self.generate_example_test()]

        # Minimal shapes: single row, single feature, two classes.
        tests.append(self._make(1, 1, 2, seed=1))

        # Zero inputs: all logits are 0, loss must be exactly log(V).
        z = self._make(3, 4, 5, seed=2)
        z["X"].zero_()
        tests.append(z)

        # Target at the last vocab index: catches off-by-one in the vocab loop / tail tile.
        tests.append(self._make(4, 16, 100, seed=3, targets=[99, 99, 0, 50]))

        # Negative-only inputs and weights.
        neg = self._make(16, 32, 64, seed=4)
        neg["X"].abs_().neg_()
        tests.append(neg)

        # Very large logits (|logit| ~ 100s): exp() without max-subtraction overflows to inf.
        tests.append(self._make(8, 64, 255, scale=40.0, seed=5))

        # Vocabulary not a multiple of any tile size and D not a power of two.
        tests.append(self._make(30, 100, 1000, seed=6))

        # Power-of-two shapes.
        tests.append(self._make(128, 256, 512, seed=7))

        # Realistic: vocab chunking across many tiles, many rows.
        tests.append(self._make(512, 512, 8192, seed=8))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._make(4096, 1024, 32768, seed=0)
