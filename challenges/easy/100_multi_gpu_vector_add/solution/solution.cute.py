import torch
import torch.distributed as dist


# For multi-gpu challenges, the runner passes torch tensors (so dist collectives work).
# Inner CuTe kernels can be written with @cute.jit and called with from_dlpack conversion.
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    chunk = (N + world_size - 1) // world_size
    lo = rank * chunk
    hi = min(N, lo + chunk)

    C.zero_()
    if lo < hi:
        # simple per-rank slice compute; a real cute kernel would live here.
        C[lo:hi] = A[lo:hi] + B[lo:hi]

    dist.all_reduce(C, op=dist.ReduceOp.SUM)
