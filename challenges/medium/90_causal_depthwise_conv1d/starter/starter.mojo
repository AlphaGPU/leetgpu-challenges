from gpu.host import DeviceContext
from memory import UnsafePointer

# x, weight, bias, output are device pointers
@export
def solve(
    x: UnsafePointer[Float32],
    weight: UnsafePointer[Float32],
    bias: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    B: Int32,
    L: Int32,
    D: Int32,
    K: Int32,
):
    pass
