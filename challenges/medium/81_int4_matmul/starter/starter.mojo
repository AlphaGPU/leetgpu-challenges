from gpu.host import DeviceContext
from memory import UnsafePointer

# x, scales, y are device pointers
@export
def solve(
    x: UnsafePointer[Float16],
    w_q: UnsafePointer[UInt8],
    scales: UnsafePointer[Float16],
    y: UnsafePointer[Float16],
    M: Int32,
    N: Int32,
    K: Int32,
    group_size: Int32,
):
    pass
