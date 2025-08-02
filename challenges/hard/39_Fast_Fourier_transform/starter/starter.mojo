from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# signal & spectrum are device-resident flat arrays of float32
@export
def solve(
    signal: UnsafePointer[Float32],
    spectrum: UnsafePointer[Float32],
    N: Int32
):
    # TODO: implement GPU FFT here
    pass
