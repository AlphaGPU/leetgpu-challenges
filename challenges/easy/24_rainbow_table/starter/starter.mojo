from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn fnv1a_hash(input: Int32) -> UInt32:
    alias FNV_PRIME: UInt32 = 16777619
    alias OFFSET_BASIS: UInt32 = 2166136261
    
    var hash: UInt32 = OFFSET_BASIS
    
    for byte_pos in range(4):
        var byte_val: UInt32 = UInt32((input >> (byte_pos * 8)) & 0xFF)
        hash = (hash ^ byte_val) * FNV_PRIME
    
    return hash

fn fnv1a_hash_kernel(input: UnsafePointer[Int32], output: UnsafePointer[UInt32], 
                     N: Int32, R: Int32):
    pass

# input, output are device pointers (i.e. pointers to memory on the GPU)
@export                         
def solve(input: UnsafePointer[Int32], output: UnsafePointer[UInt32], N: Int32, R: Int32):
    var threadsPerBlock: Int32 = 256
    var ctx = DeviceContext()
    
    var blocksPerGrid = ceildiv(N, threadsPerBlock)
    
    ctx.enqueue_function[fnv1a_hash_kernel](
        input, output, N, R,
        grid_dim = blocksPerGrid,
        block_dim = threadsPerBlock
    )
    
    ctx.synchronize() 