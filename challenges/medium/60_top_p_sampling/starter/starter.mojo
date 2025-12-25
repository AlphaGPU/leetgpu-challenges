from gpu.host import DeviceContext
from memory import UnsafePointer

@export
def solve(logits: UnsafePointer[Float32], p: UnsafePointer[Float32],
         seed: UnsafePointer[Int32], sampled_token: UnsafePointer[Int32],
         vocab_size: Int32):
    pass
