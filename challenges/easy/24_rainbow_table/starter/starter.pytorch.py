import torch

def fnv1a_hash(x: torch.Tensor) -> torch.Tensor:
    FNV_PRIME = 16777619
    OFFSET_BASIS = 2166136261
    x_int = x.to(torch.int64)
    hash_val = torch.full_like(x_int, OFFSET_BASIS, dtype=torch.int64)
    
    for byte_pos in range(4):
        byte = (x_int >> (byte_pos * 8)) & 0xFF
        hash_val = (hash_val ^ byte) * FNV_PRIME
        hash_val = hash_val & 0xFFFFFFFF
        
    return hash_val.to(torch.int32)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, R: int):
    pass 