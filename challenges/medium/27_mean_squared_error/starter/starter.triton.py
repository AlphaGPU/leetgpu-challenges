import triton
import triton.language as tl

# predictions_ptr, targets_ptr, mse_ptr are raw device pointers
def solve(predictions_ptr: int, targets_ptr: int, mse_ptr: int, N: int):
    pass 