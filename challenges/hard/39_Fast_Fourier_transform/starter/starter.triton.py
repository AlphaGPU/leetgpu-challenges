# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

# signal and spectrum are device pointers
def solve(signal_ptr: int, spectrum_ptr: int, N: int):
    pass