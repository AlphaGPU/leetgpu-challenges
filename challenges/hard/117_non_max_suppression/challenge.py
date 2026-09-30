import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Non-Maximum Suppression"
    atol = 0.0
    rtol = 0.0
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        keep: torch.Tensor,
        N: int,
        iou_threshold: float,
    ):
        """
        Greedy non-maximum suppression.

        Boxes are visited in order of decreasing score (ties broken towards the smaller
        index). A box that has not been suppressed yet is kept, and every box later in
        that order whose IoU with it is strictly greater than iou_threshold is suppressed.
        """
        assert boxes.shape == (N, 4), f"Expected boxes.shape=({N}, 4), got {boxes.shape}"
        assert scores.shape == (N,), f"Expected scores.shape=({N},), got {scores.shape}"
        assert keep.shape == (N,), f"Expected keep.shape=({N},), got {keep.shape}"
        assert boxes.dtype == torch.float32
        assert scores.dtype == torch.float32
        assert keep.dtype == torch.int32

        order = torch.argsort(scores, descending=True, stable=True)
        b = boxes.index_select(0, order)
        areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        rank = torch.arange(N, device=b.device)

        # suppress[i, j] is True when the box at rank i suppresses the box at rank j
        suppress = torch.zeros((N, N), dtype=torch.bool, device=b.device)
        rows = max(1, 4194304 // N)
        for start in range(0, N, rows):
            end = min(start + rows, N)
            lt = torch.maximum(b[start:end, None, :2], b[None, :, :2])
            rb = torch.minimum(b[start:end, None, 2:], b[None, :, 2:])
            wh = (rb - lt).clamp(min=0)
            inter = wh[..., 0] * wh[..., 1]
            iou = inter / (areas[start:end, None] + areas[None, :] - inter)
            suppress[start:end] = (iou > iou_threshold) & (rank[start:end, None] < rank[None, :])

        # A box survives when no surviving earlier box suppresses it. Iterating this rule
        # from "everything survives" reaches the greedy answer: after k rounds the first k
        # ranks are final, so the fixed point is hit in at most N rounds (a handful here).
        alive = torch.ones(N, dtype=torch.bool, device=b.device)
        for _ in range(N):
            nxt = ~(suppress & alive[:, None]).any(dim=0)
            if torch.equal(nxt, alive):
                break
            alive = nxt

        result = torch.zeros(N, dtype=torch.int32, device=b.device)
        result[order] = alive.to(torch.int32)
        keep.copy_(result)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "boxes": (ctypes.POINTER(ctypes.c_float), "in"),
            "scores": (ctypes.POINTER(ctypes.c_float), "in"),
            "keep": (ctypes.POINTER(ctypes.c_int), "out"),
            "N": (ctypes.c_int, "in"),
            "iou_threshold": (ctypes.c_float, "in"),
        }

    def _make_case(
        self,
        boxes: List[List[float]],
        scores: List[float],
        iou_threshold: float,
    ) -> Dict[str, Any]:
        N = len(scores)
        return {
            "boxes": torch.tensor(boxes, device=self.device, dtype=torch.float32),
            "scores": torch.tensor(scores, device=self.device, dtype=torch.float32),
            "keep": torch.zeros(N, device=self.device, dtype=torch.int32),
            "N": N,
            "iou_threshold": iou_threshold,
        }

    def _make_random_case(
        self,
        N: int,
        iou_threshold: float,
        seed: int,
        extent: int = 1000,
        per_object: int = 8,
        size: int = 40,
        jitter: int = 10,
    ) -> Dict[str, Any]:
        """Detector-like candidates: clusters of jittered boxes around random objects."""
        torch.manual_seed(seed)
        num_objects = max(1, N // per_object)
        centers = torch.randint(0, extent, (num_objects, 2), device=self.device)
        assignment = torch.randint(0, num_objects, (N,), device=self.device)
        corner = centers[assignment] + torch.randint(
            -jitter, jitter + 1, (N, 2), device=self.device
        )
        wh = size + torch.randint(-jitter, jitter + 1, (N, 2), device=self.device)
        corner = corner.to(torch.float32)
        boxes = torch.cat([corner, corner + wh.to(torch.float32)], dim=1)
        scores = torch.rand(N, device=self.device, dtype=torch.float32)
        return {
            "boxes": boxes,
            "scores": scores,
            "keep": torch.zeros(N, device=self.device, dtype=torch.int32),
            "N": N,
            "iou_threshold": iou_threshold,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        boxes = [
            [0.0, 0.0, 40.0, 40.0],
            [5.0, 5.0, 45.0, 45.0],
            [60.0, 0.0, 100.0, 40.0],
            [65.0, 5.0, 105.0, 45.0],
            [30.0, 50.0, 70.0, 90.0],
            [35.0, 55.0, 75.0, 95.0],
        ]
        scores = [0.90, 0.75, 0.95, 0.60, 0.55, 0.50]
        return self._make_case(boxes, scores, 0.5)

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Edge case: a single box is always kept
        tests.append(self._make_case([[0.0, 0.0, 10.0, 10.0]], [0.4], 0.5))

        # Edge case: two identical boxes, the lower-scoring one is suppressed
        tests.append(
            self._make_case(
                [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]],
                [0.3, 0.9],
                0.5,
            )
        )

        # Edge case: all scores zero, so ties are broken by index; box 1 is suppressed by
        # box 0 and therefore cannot suppress box 2 (chained suppression)
        tests.append(
            self._make_case(
                [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [2.0, 2.0, 12.0, 12.0]],
                [0.0, 0.0, 0.0],
                0.5,
            )
        )

        # Edge case: negative coordinates and a box that touches without overlapping
        tests.append(
            self._make_case(
                [
                    [-40.0, -40.0, -10.0, -10.0],
                    [-35.0, -35.0, -5.0, -5.0],
                    [-10.0, -10.0, 20.0, 20.0],
                    [-100.0, 30.0, -60.0, 70.0],
                ],
                [0.7, 0.2, 0.9, 0.1],
                0.5,
            )
        )

        # Power-of-2 sizes
        tests.append(self._make_random_case(16, 0.5, seed=11, extent=200, per_object=4))
        tests.append(self._make_random_case(64, 0.25, seed=12, extent=200, per_object=8))

        # Non-power-of-2 sizes
        tests.append(self._make_random_case(30, 0.875, seed=13, extent=300, per_object=6, jitter=2))
        tests.append(self._make_random_case(100, 0.5, seed=14, extent=120, per_object=25))
        tests.append(self._make_random_case(255, 0.75, seed=15, extent=600, per_object=10))

        # Realistic detector output: many candidates clustered around a few objects
        tests.append(self._make_random_case(4096, 0.375, seed=16, extent=1000, per_object=16))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._make_random_case(16384, 0.5, seed=2024, extent=2000, per_object=8)
