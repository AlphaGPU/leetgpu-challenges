import os

import torch
import torch.distributed as dist


def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    chunk = (N + world_size - 1) // world_size
    lo = rank * chunk
    hi = min(N, lo + chunk)

    C.zero_()
    if lo < hi:
        C[lo:hi] = A[lo:hi] + B[lo:hi]

    dist.all_reduce(C, op=dist.ReduceOp.SUM)
