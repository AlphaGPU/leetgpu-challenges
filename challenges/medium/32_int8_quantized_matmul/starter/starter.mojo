from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

@export
fn solve(A: UnsafePointer[Int8, MutExternalOrigin], B: UnsafePointer[Int8, MutExternalOrigin], C: UnsafePointer[Int8, MutExternalOrigin], M: Int32, N: Int32, K: Int32, scale_A: Float32, scale_B: Float32, scale_C: Float32, zero_point_A: Int32, zero_point_B: Int32, zero_point_C: Int32) raises:
    pass
