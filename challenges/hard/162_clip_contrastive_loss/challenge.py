import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "CLIP Contrastive Loss"
    # The loss is the average of two means of per-row / per-column log-sum-exp values
    # (magnitude ~log N <= ~10). Logits are bounded by |logit_scale| <= 100 and come from
    # dot products of unit vectors, so float32 accumulation-order differences stay ~1e-5
    # in the final mean. Every targeted mistake below (missing L2 normalization, a single
    # softmax direction, an off-diagonal target, no max subtraction) is off by >= 0.05.
    atol = 1e-03
    rtol = 1e-03
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        loss: torch.Tensor,
        N: int,
        D: int,
        logit_scale: float,
    ):
        assert image_embeds.shape == (N, D)
        assert text_embeds.shape == (N, D)
        assert loss.shape == (1,)
        assert image_embeds.dtype == text_embeds.dtype == loss.dtype == torch.float32

        img = image_embeds / image_embeds.norm(dim=1, keepdim=True)
        txt = text_embeds / text_embeds.norm(dim=1, keepdim=True)
        logits = logit_scale * (img @ txt.T)
        diag = torch.diagonal(logits)
        loss_i2t = (torch.logsumexp(logits, dim=1) - diag).mean()
        loss_t2i = (torch.logsumexp(logits, dim=0) - diag).mean()
        loss.copy_((0.5 * (loss_i2t + loss_t2i)).reshape(1))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "image_embeds": (ctypes.POINTER(ctypes.c_float), "in"),
            "text_embeds": (ctypes.POINTER(ctypes.c_float), "in"),
            "loss": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
            "logit_scale": (ctypes.c_float, "in"),
        }

    def _pack(self, img, txt, logit_scale):
        N, D = img.shape
        return {
            "image_embeds": img.contiguous(),
            "text_embeds": txt.contiguous(),
            "loss": torch.zeros(1, device=self.device, dtype=torch.float32),
            "N": N,
            "D": D,
            "logit_scale": logit_scale,
        }

    def _make(self, N, D, logit_scale, seed, noise=1.0):
        # Paired embeddings: text = image + noise. Matching pairs have the largest expected
        # similarity (as in a trained CLIP model); `noise` is chosen per shape so the loss is
        # O(1) nats and off-diagonal logits contribute meaningfully (not a degenerate ~0 loss).
        torch.manual_seed(seed)
        img = self._randn(N, D)
        txt = img + noise * self._randn(N, D)
        return self._pack(img, txt, logit_scale)

    def _randn(self, *shape):
        return torch.randn(*shape, device=self.device, dtype=torch.float32)

    def generate_example_test(self) -> Dict[str, Any]:
        img = torch.tensor([[1.0, 0.0], [0.0, 2.0]], device=self.device, dtype=torch.float32)
        txt = torch.tensor([[3.0, 4.0], [0.0, -1.0]], device=self.device, dtype=torch.float32)
        return self._pack(img, txt, 2.0)

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = [self.generate_example_test()]

        # N = 1, D = 1: the only logit is the diagonal, so the loss must be exactly 0.
        # Catches uninitialized accumulators and empty-reduction bugs.
        tests.append(self._make(1, 1, 10.0, seed=1))

        # logit_scale = 0: every logit is 0, so the loss must be exactly log(N).
        # This is the "zero input" case (zero embeddings are excluded since rows are normalized).
        tests.append(self._make(3, 4, 0.0, seed=2))

        # Rows with wildly different magnitudes (x0.01 .. x100) and N != D. Catches a
        # missing L2 normalization and normalizing along the wrong axis.
        torch.manual_seed(3)
        img = self._randn(30, 100) * torch.logspace(-2, 2, 30, device=self.device).unsqueeze(1)
        txt = self._randn(30, 100) * torch.logspace(2, -2, 30, device=self.device).unsqueeze(1)
        tests.append(self._pack(img, txt, 20.0))

        # Asymmetric logits: all image rows are nearly the same direction while text rows
        # are diverse, so the image->text and text->image losses differ by several nats.
        # Catches computing only the row (or only the column) softmax, or assuming the
        # logit matrix is symmetric.
        torch.manual_seed(4)
        base = self._randn(1, 64)
        img = base + 0.05 * self._randn(48, 64)
        txt = self._randn(48, 64) + 0.5 * base
        tests.append(self._pack(img, txt, 30.0))

        # Large logits: every embedding points close to one shared direction, so with
        # logit_scale = 100 (CLIP's clamp value) all logits lie in ~[98, 100]. exp(>88.7)
        # overflows float32, so this catches a missing max subtraction; the loss stays
        # non-trivial because the logits are tightly packed.
        torch.manual_seed(5)
        base = self._randn(1, 64)
        img = base + 0.1 * self._randn(64, 64)
        txt = img + 0.05 * self._randn(64, 64)
        tests.append(self._pack(img, txt, 100.0))

        # All-negative image entries, tiny D and N = 255 (tail tiles in both dimensions).
        neg = self._make(255, 3, 15.0, seed=6)
        neg["image_embeds"].abs_().neg_()
        tests.append(neg)

        # Duplicate text rows (repeated captions): tied columns in the logit matrix.
        torch.manual_seed(7)
        img = self._randn(16, 32)
        txt = self._randn(4, 32).repeat(4, 1)
        tests.append(self._pack(img, txt, 10.0))

        # Power-of-two shapes.
        tests.append(self._make(512, 256, 50.0, seed=8, noise=5.0))

        # Non-power-of-two shapes: N and D are not multiples of common tile sizes.
        tests.append(self._make(1000, 100, 100.0, seed=9, noise=2.5))

        # Realistic CLIP batch shard.
        tests.append(self._make(4096, 512, 100.0, seed=10, noise=5.0))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._make(16384, 512, 100.0, seed=0, noise=5.0)
