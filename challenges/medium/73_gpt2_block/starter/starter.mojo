from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# x, output, ln1_weight, ln1_bias, W_qkv, b_qkv, W_attn_proj, b_attn_proj, ln2_weight, ln2_bias, W_fc, b_fc, W_proj, b_proj are device pointers
@export
def solve(x: UnsafePointer[Float32], output: UnsafePointer[Float32], ln1_weight: UnsafePointer[Float32], ln1_bias: UnsafePointer[Float32], W_qkv: UnsafePointer[Float32], b_qkv: UnsafePointer[Float32], W_attn_proj: UnsafePointer[Float32], b_attn_proj: UnsafePointer[Float32], ln2_weight: UnsafePointer[Float32], ln2_bias: UnsafePointer[Float32], W_fc: UnsafePointer[Float32], b_fc: UnsafePointer[Float32], W_proj: UnsafePointer[Float32], b_proj: UnsafePointer[Float32], seq_len: Int32, d_model: Int32, n_heads: Int32, ffn_dim: Int32):
    pass
