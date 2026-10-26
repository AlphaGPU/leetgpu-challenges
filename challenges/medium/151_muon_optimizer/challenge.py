import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase, OutTensor, RandnTensor


class Challenge(ChallengeBase):
    name = "Muon Optimizer"
    # The Newton-Schulz iteration is a chain of 15 matrix multiplications
    # (3 per step), and its result is scaled by up to sqrt(max(M, N)). On CPU
    # the float32 reference (with its two-level Frobenius-norm sum) matches a
    # float64 implementation of the same iteration (float64 norm included) to
    # within 1.9e-5 max abs difference: at most 1.9e-5 on the functional
    # cases (worst case: the large 1024 x 512 one) and 1.3e-5 to 1.8e-5 on
    # the 1024 x 1024 performance case over five seeds. Against atol alone
    # that is up to 0.19 (2e-5 / 1e-4 = 0.2); against the elementwise bound
    # atol + rtol * |ref| (torch.allclose-style) it is at most ~0.075.
    # Reordered float32 kernels with an accurate norm measure up to ~0.14 of
    # that bound, so correct kernels pass with a ~7x margin. Every plausible
    # bug variant (wrong polynomial, no Frobenius normalization, missing
    # epsilon, wrong sqrt scale, sign flip, fewer steps) misses by orders of
    # magnitude more than this tolerance.
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        G: torch.Tensor,
        Buf: torch.Tensor,
        output: torch.Tensor,
        M: int,
        N: int,
        momentum: float,
    ):
        assert G.shape == (M, N)
        assert Buf.shape == (M, N)
        assert output.shape == (M, N)
        assert G.dtype == Buf.dtype == output.dtype == torch.float32

        # Update direction: heavy-ball combination of the momentum buffer and
        # the current gradient. Buf is a read-only input here; combining it
        # with G and writing the updated buffer back is the caller's job.
        V = momentum * Buf + G

        # Orthogonalize V with 5 steps of the quintic Newton-Schulz iteration
        # (Keller Jordan's coefficients). After the Frobenius normalization
        # the largest singular value of X is at most 1 (sigma_max <= ||X||_F
        # = 1), and each step applies the polynomial p(s) = a*s + b*s^3 +
        # c*s^5 to every singular value. The quintic has no fixed point at
        # s = 1 (p(1) = 0.701): it is tuned to lift small singular values
        # quickly. For rectangular inputs, the singular values of X after 5
        # steps land in a band around 1 (roughly [0.68, 1.13]). Square
        # Gaussian inputs behave differently: they have singular values
        # near 0 that 5 steps do not lift (at 1024 x 1024 the smallest
        # measured 0.001-0.02, the largest 1.20), so in general X is only an
        # approximation of the orthogonal polar factor of V — by design,
        # exactly as in Keller Jordan's and PyTorch's Muon implementations.
        # The 5-step iteration itself, not the exact polar factor, defines
        # the result.
        a, b, c = 3.4445, -4.7750, 2.0315
        # Frobenius norm as a two-level float32 sum (per row, then over rows).
        # A single flat float32 reduction carries ~1e-5 relative error at
        # 1024 x 1024, which the iteration passes through almost 1:1 and which
        # would penalize kernels that compute the norm more accurately.
        norm = V.pow(2).sum(dim=1).sum().sqrt()
        X = V / (norm + 1e-7)
        transposed = M > N
        if transposed:
            # Iterate with the Gram matrix on the smaller dimension.
            X = X.t()
        for _ in range(5):
            A = torch.matmul(X, X.t())
            B = b * A + c * torch.matmul(A, A)
            X = a * X + torch.matmul(B, X)
        if transposed:
            X = X.t()

        # Update scaling (this challenge's convention): a semi-orthogonal
        # factor has RMS 1 / sqrt(max(M, N)), so multiplying by sqrt(max(M,
        # N)) gives the update RMS 1. Other Muon implementations scale
        # differently — PyTorch's Muon uses 0.2 * sqrt(max(M, N)) and Keller
        # Jordan's current code uses sqrt(max(1, M / N)) — but sqrt(max(M,
        # N)) is the spec here.
        output.copy_(X * (max(M, N) ** 0.5))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "G": (ctypes.POINTER(ctypes.c_float), "in"),
            "Buf": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "momentum": (ctypes.c_float, "in"),
        }

    def _make_test_case(
        self,
        M: int,
        N: int,
        momentum: float = 0.9,
        seed: int = 0,
        zero_grad: bool = False,
        near_low_rank: bool = False,
        zero_all: bool = False,
        tiny: bool = False,
    ) -> Dict[str, Any]:
        dtype = torch.float32
        torch.manual_seed(seed)
        if near_low_rank:
            # Near-low-rank update direction: G and Buf share the same
            # singular directions (orthonormal factors from a seeded QR) —
            # one dominant direction of weight 1 and every other direction
            # weighted 0.05 in both, so V = momentum * Buf + G has
            # sigma1 = 1.9 with all remaining singular values at 0.095:
            # sigma2 / sigma1 = 0.05 (measured). The 5%-strength directions
            # are genuine signal the polynomial must lift toward the band,
            # not noise to threshold away.
            k = min(M, N)
            U, _ = torch.linalg.qr(torch.randn(M, k, device=self.device, dtype=dtype))
            W, _ = torch.linalg.qr(torch.randn(N, k, device=self.device, dtype=dtype))
            weights = torch.full((k,), 0.05, device=self.device, dtype=dtype)
            weights[0] = 1.0
            G = (U * weights) @ W.t()
            Buf = (U * weights) @ W.t()
        elif zero_all:
            G = torch.zeros(M, N, device=self.device, dtype=dtype)
            Buf = torch.zeros(M, N, device=self.device, dtype=dtype)
        elif tiny:
            # Entries of magnitude ~1e-8: ||V||_F is then ~4e-7, the same
            # order as the 1e-7 epsilon in the normalization, so whether (and
            # how) the epsilon is implemented changes the result measurably.
            G = 1e-8 * torch.randn(M, N, device=self.device, dtype=dtype)
            Buf = torch.zeros(M, N, device=self.device, dtype=dtype)
        elif zero_grad:
            G = torch.zeros(M, N, device=self.device, dtype=dtype)
            Buf = torch.randn(M, N, device=self.device, dtype=dtype)
        else:
            G = torch.randn(M, N, device=self.device, dtype=dtype)
            Buf = torch.randn(M, N, device=self.device, dtype=dtype)
        return {
            "G": G,
            "Buf": Buf,
            # Pre-filled with 123.0 so a kernel that skips entries fails instead
            # of passing on leftover zeros.
            "output": torch.full((M, N), 123.0, device=self.device, dtype=dtype),
            "M": M,
            "N": N,
            "momentum": momentum,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        G = torch.tensor(
            [[0.5, -1.0, 2.0], [1.0, 0.0, -0.5]],
            device=self.device,
            dtype=dtype,
        )
        Buf = torch.tensor(
            [[0.1, 0.2, 0.0], [-0.3, 0.4, 0.1]],
            device=self.device,
            dtype=dtype,
        )
        return {
            "G": G,
            "Buf": Buf,
            "output": torch.full((2, 3), 123.0, device=self.device, dtype=dtype),
            "M": 2,
            "N": 3,
            "momentum": 0.9,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests: List[Dict[str, Any]] = []

        # 1: the example — small literal values that can be checked by hand
        # against an independent implementation.
        tests.append(self.generate_example_test())

        # 2: square baseline.
        tests.append(self._make_test_case(64, 64, momentum=0.9, seed=42))

        # 3: wide matrix (M < N, no transpose). Catches kernels that assume
        # square shapes or always transpose.
        tests.append(self._make_test_case(128, 512, momentum=0.85, seed=1))

        # 4: tall matrix (M > N) — the transpose path, where the Gram matrix
        # must be formed on the N dimension. Catches missing or one-sided
        # transpose handling.
        tests.append(self._make_test_case(512, 128, momentum=0.95, seed=2))

        # 5: ragged shape, neither dimension a power of two or a multiple of
        # typical tile sizes.
        tests.append(self._make_test_case(37, 101, momentum=0.9, seed=3))

        # 6: small square, below most tile sizes.
        tests.append(self._make_test_case(8, 8, momentum=0.85, seed=4))

        # 7: near-low-rank update direction (sigma2 / sigma1 of V is 0.05,
        # measured). Catches implementations that threshold small singular
        # values away or diverge on ill-conditioned inputs.
        tests.append(self._make_test_case(32, 48, momentum=0.9, seed=5, near_low_rank=True))

        # 8: zero gradient — the update is the orthogonalized momentum buffer
        # alone. Catches implementations that ignore Buf.
        tests.append(self._make_test_case(48, 32, momentum=0.9, seed=6, zero_grad=True))

        # 9: momentum = 0 — the update is the orthogonalized gradient alone.
        # Catches implementations that ignore G or hardcode a momentum value.
        tests.append(self._make_test_case(32, 48, momentum=0.0, seed=7))

        # 10: larger realistic weight-matrix shape (attention projection size).
        tests.append(self._make_test_case(256, 256, momentum=0.9, seed=8))

        # 11: a single row — V has one singular value, which the Frobenius
        # normalization sets to (essentially) 1; the polynomial orbit then
        # ends at ~0.696, not 1 (p has no fixed point at 1), so the output is
        # the unit row rescaled by ~0.696 * sqrt(N). Catches dimension
        # assumptions and implementations that assume the iteration fully
        # orthonormalizes.
        tests.append(self._make_test_case(1, 16, momentum=0.9, seed=9))

        # 12: a single column — the extreme transpose path (the Gram matrix
        # is 1 x 1 after transposing). Same single-singular-value orbit as
        # the row case, ending at ~0.696, with max(M, N) = M setting the
        # scale.
        tests.append(self._make_test_case(16, 1, momentum=0.95, seed=10))

        # 13: all-zero input — V = 0, so the output must be exactly zero.
        # The 1e-7 epsilon in the normalization is all that prevents
        # 0 / 0 = NaN here. Catches a missing epsilon.
        tests.append(self._make_test_case(16, 16, momentum=0.9, seed=11, zero_all=True))

        # 14: tiny-magnitude gradient (entries ~1e-8, Buf = 0) — ||V||_F is
        # ~4e-7, so the 1e-7 epsilon shifts the normalized matrix by ~20%.
        # Catches a missing or wrong epsilon that the all-zero case alone
        # would only expose through NaN.
        tests.append(self._make_test_case(32, 48, momentum=0.9, seed=12, tiny=True))

        # 15: large rectangular shape (tall) — the transpose path at real
        # weight-matrix scale, so the general path is covered for
        # correctness at the same size class as the performance test.
        tests.append(self._make_test_case(1024, 512, momentum=0.9, seed=13))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # A square 1024 x 1024 weight matrix (an MLP projection in a ~1B
        # model). The Newton-Schulz iteration is dominated by its matrix
        # multiplications, so this shape is matmul-bound, like a real Muon
        # training step.
        return {
            "G": RandnTensor((1024, 1024)),
            "Buf": RandnTensor((1024, 1024)),
            "output": OutTensor((1024, 1024)),
            "M": 1024,
            "N": 1024,
            "momentum": 0.9,
        }
