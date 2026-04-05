import os

import torch
import torch.distributed as dist


# A, B, C are tensors on the GPU. The same inputs are replicated across all ranks.
# Each rank must leave the full result in C.
#
# The process group is already initialized by the runner (NCCL backend). Use
# torch.distributed collectives to parallelize work across ranks.
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Split [0, N) into world_size contiguous chunks and compute the local slice.
    chunk = (N + world_size - 1) // world_size
    lo = rank * chunk
    hi = min(N, lo + chunk)

    C.zero_()
    if lo < hi:
        C[lo:hi] = A[lo:hi] + B[lo:hi]

    # Combine partial results so every rank holds the full C.
    dist.all_reduce(C, op=dist.ReduceOp.SUM)
