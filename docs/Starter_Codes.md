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
- Grid and block size 
- 


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


## Easy Problems

### CUDA Starter Template

```cuda
#include <cuda_runtime.h>

__global__ void kernel_name() {
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(input, output,size) {
    
    // define grid, block size
    kernel_name<<<blocksPerGrid, threadsPerBlock>>>(input, output, size);
    cudaDeviceSynchronize();
}
```





### Triton Starter Template

```python
# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

@triton.jit
def kernel_name(input_ptr, output_ptr, input size, block size):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32)) 
    
    # TODO: Implement kernel logic
    # Use tl.program_id(0) to get block index
    # Use tl.program_id(1) to get thread ndex within block

# input_ptr, output_ptr are raw device pointers
def solve(input_ptr, output_ptr, input size):    
    # define grid, block size
    kernel_name[grid](input_ptr, output_ptr, input size, block size)
```





### Mojo Starter Template

```mojo
from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn kernel_name(input, output, size):
    # TODO: Implement kernel logic
    # Use thread_idx() to get thread index within block
    # Use block_idx() to get block index
    pass

# input, output are device pointers (i.e. pointers to memory on the GPU)
@export
def solve(input, output, size):
    #calculate threads per block
    var ctx = DeviceContext()

    ctx.enqueue_function[kernel_name](
        input, output, size,
        grid_dim = num_blocks,
        block_dim = BLOCK_SIZE
    )

    ctx.synchronize()
```

### PyTorch Starter Template

```python
import torch

def solve(input, output, size):
    # TODO: Implement solution using PyTorch operations
    pass
```

### TinyGrad Starter Template

```python
import tinygrad

def solve(input, output, size):
    # TODO: Implement solution using TinyGrad operations
    pass
```


## Medium and Hard Problems

### CUDA Starter Template

```cuda
#include <cuda_runtime.h>

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(input, output, size) {
    
}
```

### Triton Starter Template

```python
# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

# input_ptr, output_ptr are raw device pointers
def solve():    
    pass
```


### Mojo Starter Template

```mojo
from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

@export
def solve(input, output, size):

    pass
```

### PyTorch Starter Template

```python
import torch

def solve(input, output, size):
    # TODO: Implement solution using PyTorch operations
    pass
```

### TinyGrad Starter Template

```python
import tinygrad

def solve(input, output, size):
    # TODO: Implement solution using TinyGrad operations
    pass
```
