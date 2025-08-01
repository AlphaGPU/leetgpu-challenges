# Starter Code Creation Process for LeetGPU Challenges

This guide explains the complete process of creating starter codes for LeetGPU challenges, from understanding the requirements to implementing across all frameworks.


## Analyzing Challenge Requirements

###Identify Framework Requirements

Each framework has specific requirements:

**CUDA:**
- Kernel functions with `__global__` qualifier
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
- Type hints for clarity

## Designing Function Signatures

### Step 1: Define Core Parameters

Based on the algorithm requirements, determine:

1. **Input parameters**: What data does the algorithm need?
2. **Output parameters**: Where should results be written?
3. **Size parameters**: What dimensions are involved?
4. **Configuration parameters**: Any algorithm-specific settings?

### Step 2: Choose Parameter Names

Use clear, descriptive names that follow conventions:

**Common Patterns:**
- Single input/output: `input`, `output`
- Multiple inputs: `A`, `B`, `C` or `Q`, `K`, `V`
- Dimensions: `N`, `M`, `K`, `rows`, `cols`
- Algorithm-specific: `kernel_size`, `stride`, `padding`

### Step 3: Determine Data Types

**Standard Types:**
- **CUDA**: `const float*` for inputs, `float*` for outputs, `int` for sizes
- **Triton**: `int` for pointers, `int` for sizes
- **Mojo**: `UnsafePointer[Float32]` for data, `Int32` for sizes
- **PyTorch/TinyGrad**: `Tensor` for data, `int` for sizes

### Step 4: Create Function Signatures

**Example: ReLU Activation**

```cuda
// CUDA
extern "C" void solve(const float* input, float* output, int N)

// Triton
def solve(input_ptr: int, output_ptr: int, N: int)

// Mojo
@export
def solve(input: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32)

// PyTorch
def solve(input: torch.Tensor, output: torch.Tensor, N: int)
```

## Implementing Across Frameworks

### Step 1: CUDA Implementation

**Basic Structure:**
```cuda
#include <cuda_runtime.h>

__global__ void kernel_name(const float* input, float* output, int N) {
    // TODO: Implement kernel logic
}

extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    kernel_name<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

### Step 2: Triton Implementation

**Basic Structure:**
```python
# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

@triton.jit
def kernel_name(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))
    
def solve(input_ptr: int, output_ptr: int, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    kernel_name[grid](input_ptr, output_ptr, N, BLOCK_SIZE)
```

### Step 3: Mojo Implementation

**Basic Structure:**
```mojo
from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn kernel_name(input: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32):
    pass

@export
def solve(input: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32):
    pass
```

### Step 4: PyTorch Implementation

**Basic Structure:**
```python
import torch

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    pass
```

### Step 5: TinyGrad Implementation

**Basic Structure:**
```python
import tinygrad

def solve(input: tinygrad.Tensor, output: tinygrad.Tensor, N: int):
    pass
```

-----

A proper starter code gives participants a runnable foundation while keeping the solution logic as their task.



