# PyTorch is not allowed inside Triton solutions for benchmarking fairness.
import triton
import triton.language as tl

# signal_ptr and spectrum_ptr are raw device pointers (int)
def solve(signal_ptr: int, spectrum_ptr: int, N: int):
    # TODO: write one (or several) Triton kernels that perform the FFT
    pass
