from gpu.host import DeviceContext
from memory import UnsafePointer
from gpu.id import block_dim, block_idx, thread_idx

@export
def solve(logits: UnsafePointer[Float32], p: UnsafePointer[Float32],
         top_p_probs: UnsafePointer[Float32],
         vocab_size: Int32):
    pass
