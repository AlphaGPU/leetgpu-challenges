from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

@export
def solve(A: UnsafePointer[Int8], B: UnsafePointer[Int8], C: UnsafePointer[Int8], M: Int32, N: Int32, K: Int32, scale_A: Float32, scale_B: Float32, scale_C: Float32, zero_point_A: Int32, zero_point_B: Int32, zero_point_C: Int32):
    pass 