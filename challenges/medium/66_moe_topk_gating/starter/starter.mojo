from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

@export
def solve(logits: UnsafePointer[Float32], topk_weights: UnsafePointer[Float32], topk_indices: UnsafePointer[Int32], M: Int32, E: Int32, k: Int32):
    pass
