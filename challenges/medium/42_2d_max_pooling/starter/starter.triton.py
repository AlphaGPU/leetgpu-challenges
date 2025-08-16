import torch
import triton
import triton.language as tl

# input, output are tensors on the GPU
def solve(input, output, N, C, H, W, kernel_size, stride, padding):
   pass