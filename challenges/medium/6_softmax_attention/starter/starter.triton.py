# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

# Q_ptr, K_ptr, V_ptr, output_ptr are raw device pointers
def solve(Q_ptr: int, K_ptr: int, V_ptr: int, output_ptr: int, M: int, N: int, d: int):
    pass 