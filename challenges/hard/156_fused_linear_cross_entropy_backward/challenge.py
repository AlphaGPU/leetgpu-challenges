import ctypes
from typing import Any, Dict, List, Optional

import torch
from core.challenge_base import ChallengeBase

IGNORE_INDEX = -100


class Challenge(ChallengeBase):
    name = "Fused Linear Cross Entropy Backward"
    # Gradients of a mean over n_valid rows are O(|W| / n_valid) for dX and O(|X| / n_valid) for
    # dW, so the bound is dominated by rtol * max|ref| (atol only matters when an output is all
    # zero, e.g. every target ignored or X = 0, where it must be exactly 0). Against a float64
    # recomputation the float32 reference and independent float32 orderings (chunked online
    # softmax, split-V accumulation) stay under 5% of the atol + rtol * max|ref| budget on every
    # case. Rounding X and W to FP16 before the matmuls misses by ~2x and BF16 by ~15x; dividing by
    # N instead of the number of non-ignored rows, giving ignored rows a gradient, dropping the
    # one-hot term or a vocabulary tail, or a non-rescaled online softmax miss by far more.
    atol = 1e-08
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        targets: torch.Tensor,
        dX: torch.Tensor,
        dW: torch.Tensor,
        N: int,
        D: int,
        V: int,
    ):
        assert X.shape == (N, D), f"Expected X.shape=({N}, {D}), got {X.shape}"
        assert W.shape == (V, D), f"Expected W.shape=({V}, {D}), got {W.shape}"
        assert targets.shape == (N,), f"Expected targets.shape=({N},), got {targets.shape}"
        assert dX.shape == (N, D), f"Expected dX.shape=({N}, {D}), got {dX.shape}"
        assert dW.shape == (V, D), f"Expected dW.shape=({V}, {D}), got {dW.shape}"
        assert X.dtype == W.dtype == dX.dtype == dW.dtype == torch.float32
        assert targets.dtype == torch.int32

        valid = targets != IGNORE_INDEX
        safe_t = torch.where(valid, targets, torch.zeros_like(targets)).long()
        n_valid = valid.sum().clamp(min=1).to(torch.float32)
        row_scale = valid.to(torch.float32) / n_valid

        # G = (softmax(X W^T) - onehot(targets)) * valid / n_valid, built in place.
        G = torch.softmax(torch.matmul(X, W.t()), dim=1)
        G.scatter_add_(1, safe_t.unsqueeze(1), torch.full((N, 1), -1.0, device=X.device))
        G.mul_(row_scale.unsqueeze(1))

        dX.copy_(torch.matmul(G, W))
        dW.copy_(torch.matmul(G.t(), X))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "X": (ctypes.POINTER(ctypes.c_float), "in"),
            "W": (ctypes.POINTER(ctypes.c_float), "in"),
            "targets": (ctypes.POINTER(ctypes.c_int), "in"),
            "dX": (ctypes.POINTER(ctypes.c_float), "out"),
            "dW": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
            "V": (ctypes.c_int, "in"),
        }

    def _make(
        self,
        N: int,
        D: int,
        V: int,
        seed: int = 0,
        scale: float = 1.0,
        targets: Optional[List[int]] = None,
        ignore_frac: float = 0.0,
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        dev, f32 = self.device, torch.float32
        X = torch.empty(N, D, device=dev, dtype=f32).uniform_(-1.0, 1.0)
        W = torch.empty(V, D, device=dev, dtype=f32).uniform_(-1.0, 1.0)
        W.mul_(scale / (D**0.5))
        if targets is None:
            t = torch.randint(0, V, (N,), device=dev, dtype=torch.int32)
            if ignore_frac > 0.0:
                t[torch.rand(N, device=dev) < ignore_frac] = IGNORE_INDEX
        else:
            t = torch.tensor(targets, device=dev, dtype=torch.int32)
        return {
            "X": X,
            "W": W,
            "targets": t,
            # Pre-filled with garbage, as in the other backward challenges: dX and dW are
            # documented as not zero-initialized.
            "dX": torch.full((N, D), 123.0, device=dev, dtype=f32),
            "dW": torch.full((V, D), 123.0, device=dev, dtype=f32),
            "N": N,
            "D": D,
            "V": V,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dev, f32 = self.device, torch.float32
        return {
            "X": torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], device=dev, dtype=f32),
            "W": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], device=dev, dtype=f32),
            "targets": torch.tensor([0, 2, IGNORE_INDEX], device=dev, dtype=torch.int32),
            "dX": torch.full((3, 2), 123.0, device=dev, dtype=f32),
            "dW": torch.full((3, 2), 123.0, device=dev, dtype=f32),
            "N": 3,
            "D": 2,
            "V": 3,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Example from the description (row 2 ignored, n_valid = 2); its values are checked
        # against a hand-derived closed form in the problem statement.
        tests.append(self.generate_example_test())

        # Minimal: one row, one feature, two classes.
        tests.append(self._make(1, 1, 2, seed=1))

        # Every target ignored: both gradients must be exactly 0 everywhere (catches 0 / 0 = nan
        # from dividing by n_valid = 0, and outputs left unwritten).
        tests.append(self._make(4, 8, 16, seed=2, targets=[IGNORE_INDEX] * 4))

        # Ignored rows at the first, middle and last positions: catches dividing by N instead of
        # n_valid (off by 7/4) and ignored rows still contributing softmax(z) to dX and dW.
        tests.append(
            self._make(7, 16, 50, seed=3, targets=[-100, 3, 49, -100, 0, 17, IGNORE_INDEX])
        )

        # Zero activations: logits are all 0, dW must be exactly 0 and dX = (mean_v W_v -
        # W_target) / n_valid.
        zero = self._make(5, 12, 33, seed=4)
        zero["X"].zero_()
        tests.append(zero)

        # Logits of several hundred: exp() without subtracting the row max overflows to inf/nan.
        tests.append(self._make(8, 64, 255, seed=5, scale=200.0))

        # Every row has the same target, the last vocabulary index, with V not a multiple of
        # any tile size: catches races on dW[V - 1] (non-atomic one-hot updates) and a
        # vocabulary loop that drops the tail tile.
        tests.append(self._make(64, 40, 1000, seed=6, targets=[999] * 64))

        # Negative-only activations and weights.
        neg = self._make(16, 32, 64, seed=7)
        neg["X"].abs_().neg_()
        neg["W"].abs_().neg_()
        tests.append(neg)

        # Power-of-two shapes.
        tests.append(self._make(128, 256, 512, seed=8, ignore_frac=0.1))

        # Non-power-of-two shapes with large logits spread over many vocabulary tiles: an
        # online softmax that forgets to rescale the running sum when the max grows fails here.
        tests.append(self._make(30, 100, 3000, seed=9, scale=8.0, ignore_frac=0.2))

        # Realistic: many rows and a vocabulary spanning many tiles, with padding rows ignored.
        tests.append(self._make(512, 512, 8192, seed=10, ignore_frac=0.1))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # LM-head backward for 4,096 tokens, hidden size 1,024 and a 32,768-token vocabulary,
        # with ~5% padding tokens ignored.
        return self._make(4096, 1024, 32768, seed=156, ignore_frac=0.05)
