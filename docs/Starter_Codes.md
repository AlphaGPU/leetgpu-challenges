# Starter Code Creation Process for LeetGPU Challenges

A starter code is a template file that provides the basic structure and function signatures for implementing GPU-accelerated algorithms in LeetGPU challenges. It gives users a runnable foundation while leaving the core algorithmic logic as their task.

## Major Components

- **Function Signatures:** Standardized `solve` function with consistent parameters across all frameworks
- **Framework-Specific Templates:** CUDA, Triton, Mojo, PyTorch, and TinyGrad implementations
- **Memory Management:** Proper device pointer handling and memory allocation patterns
- **Kernel Structure:** Basic kernel function templates with grid/block sizing
- **Error Handling:** Bounds checking and synchronization primitives


### Identify Framework Requirements

Each framework has specific requirements:

**CUDA:**
- Kernel functions with `__global__` qualifier(for easy problems)
- `extern "C"` solve function for framework integration
- Proper memory management and synchronization
- Grid and block size calculations

**Triton:**
- `@triton.jit` decorator for kernel compilation
- Pointer type conversions for data types
- Block size and grid calculations
- PyTorch restriction compliance

**Mojo:**
- `@export` decorator for framework integration
- Proper GPU imports and memory types
- Device context management
- Function parameter types

**PyTorch/TinyGrad:**
- Tensor-based function signatures
- GPU tensor parameters
- Simple, direct implementations


Based on the algorithm requirements, determine:

1. **Input parameters**: What data does the algorithm need?
2. **Output parameters**: Where should results be written?
3. **Size parameters**: What dimensions are involved?
4. **Configuration parameters**: Any algorithm-specific settings?


## Easy Problems templates


### CUDA Starter Template

```cuda
#include <cuda_runtime.h>

__global__ void kernel_name(const float* input, float* output, int N) {
    // TODO: Implement kernel logic
    // Each thread processes one element
    // Use threadIdx.x + blockIdx.x * blockDim.x to get global index
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    kernel_name<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

### Triton Starter Template

```python
# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

@triton.jit
def kernel_name(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32)) 
    
    # TODO: Implement kernel logic
    # Use tl.program_id(0) to get block index
    # Use tl.program_id(1) to get thread ndex within block

# input_ptr, output_ptr are raw device pointers
def solve(input_ptr: int, output_ptr: int, N: int):    
    BLOCK_SIZE = 1024
    # define grid
    kernel_name[grid](input_ptr, output_ptr, N, BLOCK_SIZE)
```

### Mojo Starter Template

```mojo
from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn kernel_name(input: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32):
    # TODO: Implement kernel logic
    # Use thread_idx() to get thread index
    # Use block_idx() to get block index
    pass

@export
def solve(input: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32):
    # define block, thread size
    var ctx = DeviceContext()

    # Launch the kernel using enqueue_function
    ctx.enqueue_function[kernel_name](
        input_ptr, output_ptr, N,
        grid_dim  = num_blocks,     # Number of blocks in 1D grid
        block_dim = BLOCK_SIZE      # Number of threads per block
    )

    ctx.synchronize()
    # TODO: Implement solve function
    pass
```

### PyTorch Starter Template

```python
import torch

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    # TODO: Implement solution using PyTorch operations
    pass
```

### TinyGrad Starter Template

```python
import tinygrad

def solve(input: tinygrad.Tensor, output: tinygrad.Tensor, N: int):
    # TODO: Implement solution using TinyGrad operations
    pass
```

For medium and hard problems, only define solve function for CUDA, Mojo and Triton.