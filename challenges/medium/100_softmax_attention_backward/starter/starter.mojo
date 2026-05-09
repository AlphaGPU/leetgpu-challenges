from gpu.host import DeviceContext
from memory import UnsafePointer


# Q, K, V, dO, dQ, dK, dV are device pointers
@export
def solve(
    Q: UnsafePointer[Float32],
    K: UnsafePointer[Float32],
    V: UnsafePointer[Float32],
    dO: UnsafePointer[Float32],
    dQ: UnsafePointer[Float32],
    dK: UnsafePointer[Float32],
    dV: UnsafePointer[Float32],
    M: Int32,
    N: Int32,
    d: Int32,
):
    pass
