from gpu.host import DeviceContext
from memory import UnsafePointer

# x, residual, weight, out are device pointers
@export
def solve(
    x: UnsafePointer[Float32],
    residual: UnsafePointer[Float32],
    weight: UnsafePointer[Float32],
    out: UnsafePointer[Float32],
    N: Int32,
    C: Int32,
    eps: Float32,
):
    pass
