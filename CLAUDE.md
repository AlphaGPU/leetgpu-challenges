# LeetGPU Challenge Creation Guide

This guide provides instructions for adding new challenges to the LeetGPU challenge set, covering structure, metadata, test cases, and best practices.

## Challenge Structure

Each challenge must be placed in a specific directory hierarchy as specified [here](CONTRIBUTING.md)

### Directory Naming Convention

- **Pattern**: `<number>_<challenge_name>`
- **Number**: Sequential integer within each difficulty
- **Name**: Lowercase with underscores (e.g., `vector_add`, `matrix_multiplication`)

---

## Challenge Types & Difficulty Levels

### Easy Challenges
**Definition**: Single core concept, basic parallelization.
- 1-2 input parameters plus output
- Element-wise operations or basic matrix operations
- Clear algorithmic approach, minimal optimization
- Examples: Vector addition, matrix transposition, element-wise operations

### Medium Challenges
**Definition**: Multiple concepts, memory optimizations.
- 2-4 input/output parameters
- Memory hierarchies, reduction patterns, tiling
- Examples: Matrix multiplication with tiling, 2D convolution

### Hard Challenges
**Definition**: Advanced techniques, complex algorithms.
- Multiple parameters with complex relationships
- Advanced optimizations (warp operations, cooperative groups)
- Non-trivial algorithms, heavy performance requirements
- Examples: Optimized matrix multiplication, GPU sorting, graph algorithms

---

## Challenge.py Specification

The `challenge.py` file contains the reference implementation, test cases, and metadata. It must inherit from `ChallengeBase`.

### Class Declaration & Initialization

```python
from typing import Any, Dict, List
import torch
import ctypes
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Challenge Display Name",
            atol=1e-05,              # Absolute tolerance
            rtol=1e-05,              # Relative tolerance
            num_gpus=1,              # GPUs required
            access_tier="free"       # "free" or "premium"
        )
```

### Reference Implementation (`reference_impl`)

Must accept same parameters as user solution, perform correct computation, include input validation.

```python
def reference_impl(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    assert A.shape == B.shape == C.shape
    assert A.dtype == B.dtype == C.dtype == torch.float32
    assert A.device == B.device == C.device == torch.device('cuda')

    torch.add(A, B, out=C)
```

### Solve Signature (`get_solve_signature`)

Defines function signature users must implement.

```python
def get_solve_signature(self) -> Dict[str, tuple]:
    return {
        "A": (ctypes.POINTER(ctypes.c_float), "in"),
        "B": (ctypes.POINTER(ctypes.c_float), "in"),
        "C": (ctypes.POINTER(ctypes.c_float), "out"),
        "N": (ctypes.c_size_t, "in"),
    }
```

**Common ctypes**: `ctypes.POINTER(ctypes.c_float)`, `ctypes.c_int`, `ctypes.c_size_t`
**Parameter directions**: `"in"` (read-only), `"out"` (write-only), `"inout"` (read and write)

### Test Case Generation

#### Example Test (`generate_example_test`)
Generates one simple test case for display.

```python
def generate_example_test(self) -> Dict[str, Any]:
    N = 4
    A = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=torch.float32)
    B = torch.tensor([5.0, 6.0, 7.0, 8.0], device="cuda", dtype=torch.float32)
    C = torch.empty(N, device="cuda", dtype=torch.float32)

    return {"A": A, "B": B, "C": C, "N": N}
```

#### Functional Tests (`generate_functional_test`)
Generates 10-15 test cases covering edge cases, various sizes, special values.

```python
def generate_functional_test(self) -> List[Dict[str, Any]]:
    test_cases = []
    sizes = [1, 2, 3, 4, 8, 16, 32, 64, 100, 256, 1000, 10000]
    
    for size in sizes:
        test_cases.append({
            "A": torch.randn(size, device="cuda", dtype=torch.float32),
            "B": torch.randn(size, device="cuda", dtype=torch.float32),
            "C": torch.zeros(size, device="cuda", dtype=torch.float32),
            "N": size,
        })
    
    # Special cases: zeros, negatives
    test_cases.extend([
        {"A": torch.zeros(4, device="cuda", dtype=torch.float32), "B": torch.zeros(4, device="cuda", dtype=torch.float32), "C": torch.zeros(4, device="cuda", dtype=torch.float32), "N": 4},
        {"A": torch.tensor([-1.0, -2.0], device="cuda", dtype=torch.float32), "B": torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32), "C": torch.zeros(2, device="cuda", dtype=torch.float32), "N": 2}
    ])
Generates one large test case for benchmarking.

```python
def generate_performance_test(self) -> Dict[str, Any]:
    N = 25_000_000  # Adjust based on operation complexity
    return {
        "A": torch.empty(N, device="cuda", dtype=torch.float32).uniform_(-1000.0, 1000.0),
        "B": torch.empty(N, device="cuda", dtype=torch.float32).uniform_(-1000.0, 1000.0),
        "C": torch.zeros(N, device="cuda", dtype=torch.float32),
        "N": N,
    }
```

---

## Challenge.html Specification

The HTML file presents the problem to users as a clean fragment.

### Required Sections

#### 1. Problem Description
2-3 sentences stating what the function must do, data types, constraints.

#### 2. Implementation Requirements
- External libraries not permitted (unless required)
- Function signature must remain unchanged
- Output storage location

#### 3. Examples (1-3 minimum)
```html
<h2>Example 1:</h2>
<pre>
Input:  A = [1.0, 2.0, 3.0, 4.0]
        B = [5.0, 6.0, 7.0, 8.0]
Output: C = [6.0, 8.0, 10.0, 12.0]
</pre>
```

#### 4. Constraints
- Size constraints (min/max N)
- Data type constraints
- Value ranges

### HTML Formatting
- Use `<code>` for variables/functions
- Use `<pre>` for multi-line code/examples
- Use `&le;`, `&ge;`, `&times;` for math symbols

---

## Starter Code Guidelines

Starter code must compile without errors but not solve the problem. Follow existing comment styles exactly.

### General Principles
1. **Compilation**: Must compile/run without errors
2. **Non-functional**: Use `pass` or empty kernels
3. **Comments**: No comments outside of those displayed below. 
4. **Consistency**: Match existing starters for each framework

### Framework Examples

#### CUDA (`starter.cu`)
```cpp
#include <cuda_runtime.h>

__global__ void kernel(const float* A, const float* B, float* C, int N) {}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```

#### PyTorch (`starter.pytorch.py`)
```python
import torch

# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    pass
```

#### Triton (`starter.triton.py`)
```python
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pass

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    kernel[grid](a, b, c, N, BLOCK_SIZE)
```

#### JAX (`starter.jax.py`)
```python
import jax
import jax.numpy as jnp


# A, B are tensors on GPU
@jax.jit
def solve(A: jax.Array, B: jax.Array, N: int) -> jax.Array:
    # return output tensor directly
    pass
```


#### Other Frameworks
- **Mojo**: Use `UnsafePointer[Float32]`, `DeviceContext`, grid/block dimensions
- **CuTe**: Use `@cute.jit` decorator, `cute.Tensor` types

---

## Test Case Design

### Test Case Sizing Strategy

| Type | Size Range | Purpose | Count |
|------|-----------|---------|-------|
| Edge cases | 1-8 elements | Boundary conditions | 3-4 |
| Power-of-2 | 16-1024 elements | Common configurations | 3-4 |
| Non-power-of-2 | 30, 100, 255 | Irregular workloads | 3-4 |
| Random sizes | 1K-10K elements | Realistic sizes | 2-3 |
| **Total** | - | - | **12-15 cases** |

### Performance Test Sizing
Test case size should be limited so that 5x its size can fit comfortably within 16GB (Tesla T4 VRAM).

- **1D operations**: 10M-100M elements
- **2D operations**: 4K×4K to 8K×8K matrices
- **Complex algorithms**: 1M-10M elements (adjusted for complexity)

### Numerical Stability
Use appropriate tolerances: `atol=1e-5`, `rtol=1e-5` for float32. Avoid extreme ranges that cause overflow/underflow.

## Creating & Testing Challenges

### Manual Creation Process

1. **Create Directory Structure**
   ```bash
   mkdir -p challenges/easy/<number>_challenge_name/starter
   ```

2. **Write challenge.py**
   - Inherit from ChallengeBase
   - Implement reference_impl with assertions
   - Generate diverse test cases

3. **Write challenge.html**
   - Clear problem description
   - 1-3 examples
   - Precise constraints

4. **Write Starter Code and Test Locally**
    - Follow format and comment specifications for each framework to write starter code
    - Test:
   ```bash
   python -c "from challenges.easy.<number>_challenge_name.challenge import Challenge; c = Challenge(); print('Tests:', len(c.generate_functional_test()))"
   ```

### Validating Test Coverage
Ensure functional tests cover:
- Single element (N=1)
- Edge cases (N=2,3,4)
- Powers of 2 up to 1024
- Non-powers of 2
- Zero inputs, negative numbers, mixed values
- Large/small numbers
- Typical scales (1K-10K elements)

---

## Formatting & Linting
See [CONTRIBUTING.md](CONTRIBUTING.md)

## Directory Structure Checklist

When adding a challenge, verify:

```
✓ Directory: <number>_<name>
✓ challenge.html: description, requirements, examples, constraints
✓ challenge.py: ChallengeBase inheritance, reference_impl, signatures, test generators
✓ starter/: All framework files (cu, pytorch.py, triton.py, mojo, cute.py, jax.py)
✓ Linting: black, isort, flake8 for Python; clang-format for CUDA
✓ Tests: Functional tests pass, performance test completes in <10 seconds
```

## Example: Matrix Transpose Challenge

### challenge.html
```html
<p>Transpose a square matrix in-place on the GPU. Element [i,j] becomes [j,i].</p>

<h2>Implementation Requirements</h2>
<ul>
  <li>The <code>solve</code> function signature must remain unchanged</li>
  <li>External libraries are not permitted</li>
</ul>

<h2>Example:</h2>
<pre>
Input:  M = [[1, 2], [3, 4]]
Output: M = [[1, 3], [2, 4]]
</pre>

<h2>Constraints</h2>
<ul>
  <li>Square matrix: N×N</li>
  <li>1 ≤ N ≤ 8192</li>
</ul>
```

### challenge.py (key parts)
```python
class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(name="Matrix Transpose", atol=1e-05, rtol=1e-05, num_gpus=1, access_tier="free")

    def reference_impl(self, M: torch.Tensor, N: int):
        assert M.shape == (N, N) and M.dtype == torch.float32
        M.copy_(M.t())

    def get_solve_signature(self):
        return {"M": (ctypes.POINTER(ctypes.c_float), "inout"), "N": (ctypes.c_size_t, "in")}

    def generate_functional_test(self):
        return [{"M": torch.randn((size, size), device="cuda", dtype=torch.float32), "N": size}
                for size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]]

    def generate_performance_test(self):
        N = 4096
        return {"M": torch.randn((N, N), device="cuda", dtype=torch.float32), "N": N}
```



## Resources

- **ChallengeBase**: `core/challenge_base.py` in leetgpu-pset
- **Existing Challenges**: Browse `challenges/` directory
- **Framework Docs**: PyTorch, Triton, Mojo, JAX, CuTe

---

## Contributing

1. Fork repository
2. Create branch: `git checkout -b challenge/your-challenge-name`
3. Follow this guide
4. Run linting and tests
5. Submit PR

## Example: Matrix Transpose Challenge

**Step 1: Create structure**
```bash
mkdir -p challenges/easy/3_matrix_transpose/starter
```

**Step 2: Write challenge.html**
```html
<p>
  Implement a program that transposes a square matrix in-place on the GPU.
  Given an N×N matrix, compute the transpose where element [i,j] becomes [j,i].
</p>

<h2>Implementation Requirements</h2>
<ul>
  <li>The <code>solve</code> function signature must remain unchanged</li>
  <li>External libraries are not permitted</li>
  <li>In-place or out-of-place transpose is acceptable</li>
</ul>

<h2>Example 1:</h2>
<pre>
Input:  M = [[1, 2],
             [3, 4]]
Output: M = [[1, 3],
             [2, 4]]
</pre>

<h2>Constraints</h2>
<ul>
  <li>Input matrix is square: N×N</li>
  <li>1 ≤ N ≤ 8192</li>
  <li>Matrix elements are 32-bit floats</li>
</ul>
```

**Step 3: Write challenge.py**
```python
from typing import Any, Dict, List
import torch
import ctypes
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Matrix Transpose",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, M: torch.Tensor, N: int):
        assert M.shape == (N, N)
        assert M.dtype == torch.float32
        result = M.t()
        M.copy_(result)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "M": (ctypes.POINTER(ctypes.c_float), "inout"),
            "N": (ctypes.c_size_t, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        N = 2
        M = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=torch.float32)
        return {"M": M, "N": N}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        test_cases = []
        sizes = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        
        for size in sizes:
            M = torch.randn((size, size), device="cuda", dtype=torch.float32)
            test_cases.append({"M": M.clone(), "N": size})
        
        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        N = 4096
        M = torch.randn((N, N), device="cuda", dtype=torch.float32)
        return {"M": M, "N": N}
```

**Step 4: Starter code and validate**
Write starter code according to the rules for each framework and validate outputs. 

## Common Pitfalls & Solutions

| Issue | Solution |
|-------|----------|
| Test cases all pass but performance is terrible | Increase performance test size; check if solution is doing unnecessary work |
| Inconsistent numerical results across frameworks | Ensure tolerance values (atol/rtol) match precision capabilities |
| Starter code doesn't compile | Test locally before submitting; check imports and syntax |
| Test sizes inconsistent between easy/medium/hard | Reference this guide's sizing recommendations |
| HTML formatting looks broken | Use proper HTML entities (&le;, &ge;, &times;) |
| Reference implementation is too slow | Optimize using PyTorch kernels rather than Python loops |

## Resources & References

- **ChallengeBase**: See `core/challenge_base.py` in leetgpu-pset
- **Existing Challenges**: Browse `challenges/` for examples in each difficulty level
- **CUDA Best Practices**: Refer to NVIDIA CUDA programming guide
- **Framework Docs**:
  - PyTorch: https://pytorch.org/docs
  - Triton: https://openai.github.io/triton-docs
  - Mojo: https://docs.modular.com/mojo
  - JAX: https://jax.readthedocs.io
  - CuTe: https://github.com/NVIDIA/cutlass