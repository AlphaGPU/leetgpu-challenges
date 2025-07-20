from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

@export
def solve(input: UnsafePointer[Float32], kernel: UnsafePointer[Float32], output: UnsafePointer[Float32], input_depth: Int32, input_rows: Int32, input_cols: Int32, kernel_depth: Int32, kernel_rows: Int32, kernel_cols: Int32):
    pass 