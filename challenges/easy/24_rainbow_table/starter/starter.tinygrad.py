import tinygrad

def fnv1a_hash(x: tinygrad.Tensor) -> tinygrad.Tensor:
    FNV_PRIME = 16777619
    OFFSET_BASIS = 2166136261
    x_uint = x.cast(tinygrad.dtypes.uint32)
    hash_val = tinygrad.Tensor.full_like(x_uint, OFFSET_BASIS, dtype=tinygrad.dtypes.uint32)
    for byte_pos in range(4):
        byte = (x_uint >> (byte_pos * 8)) & 0xFF
        hash_val = ((hash_val ^ byte).cast(tinygrad.dtypes.uint64) * FNV_PRIME) & 0xFFFFFFFF
        hash_val = hash_val.cast(tinygrad.dtypes.uint32)
    return hash_val

# input, output are tensors on the GPU
def solve(input: tinygrad.Tensor, output: tinygrad.Tensor, N: int, R: int):
    pass