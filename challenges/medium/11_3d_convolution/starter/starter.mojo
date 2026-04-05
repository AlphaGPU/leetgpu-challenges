from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

@export
fn solve(input: UnsafePointer[Float32, MutExternalOrigin], kernel: UnsafePointer[Float32, MutExternalOrigin], output: UnsafePointer[Float32, MutExternalOrigin], input_depth: Int32, input_rows: Int32, input_cols: Int32, kernel_depth: Int32, kernel_rows: Int32, kernel_cols: Int32) raises:
    pass
