from gpu.host import DeviceContext
from memory import UnsafePointer

# x, W, A, B, output are device pointers
@export
def solve(
    x: UnsafePointer[Float32],
    W: UnsafePointer[Float32],
    A: UnsafePointer[Float32],
    B: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    batch: Int32,
    d_in: Int32,
    d_out: Int32,
    rank: Int32,
    lora_scale: Float32,
):
    pass
