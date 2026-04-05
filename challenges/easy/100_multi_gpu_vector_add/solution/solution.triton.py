import torch
import torch.distributed as dist
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(A, B, C, lo, hi, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = lo + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < hi
    a = tl.load(A + offs, mask=mask)
    b = tl.load(B + offs, mask=mask)
    tl.store(C + offs, a + b, mask=mask)


def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    chunk = (N + world_size - 1) // world_size
    lo = rank * chunk
    hi = min(N, lo + chunk)

    C.zero_()
    local = hi - lo
    if local > 0:
        BLOCK = 1024
        grid = (triton.cdiv(local, BLOCK),)
        vector_add_kernel[grid](A, B, C, lo, hi, BLOCK)

    dist.all_reduce(C, op=dist.ReduceOp.SUM)
