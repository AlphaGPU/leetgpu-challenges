# LeetGPU Challenge Creation Guide

This guide provides detailed instructions for adding new challenges to the LeetGPU challenge set. It covers challenge structure, metadata formatting, test case sizing, and best practices for both manual creation and AI-assisted generation.

## Table of Contents

1. [Challenge Structure](#challenge-structure)
2. [Challenge Types & Difficulty Levels](#challenge-types--difficulty-levels)
3. [Challenge.py Specification](#challengepy-specification)
4. [Challenge.html Specification](#challengehtml-specification)
5. [Starter Code Guidelines](#starter-code-guidelines)
6. [Test Case Design](#test-case-design)
7. [Creating & Testing Challenges](#creating--testing-challenges)
8. [Formatting & Linting](#formatting--linting)
9. [Directory Structure Checklist](#directory-structure-checklist)

---

## Challenge Structure

Each challenge must be placed in a specific directory hierarchy with all required files:

```
challenges/
├── easy/
│   ├── <number>_<challenge_name>/
│   │   ├── challenge.html          # Problem description (HTML)
│   │   ├── challenge.py            # Python implementation file
│   │   └── starter/                # Starter code templates
│   │       ├── starter.cu          # CUDA starter
│   │       ├── starter.pytorch.py  # PyTorch starter
│   │       ├── starter.triton.py   # Triton starter
│   │       ├── starter.mojo        # Mojo starter
│   │       ├── starter.cute.py     # CuTe starter
│   │       └── starter.jax.py      # JAX starter
│   │
│   └── 2_matrix_multiplication/
│       └── ...
│
├── medium/
│   └── ...
│
└── hard/
    └── ...
```

### Directory Naming Convention

- **Pattern**: `<number>_<challenge_name>`
- **Number**: Sequential integer (1, 2, 3, ... within each difficulty)
- **Name**: Lowercase with underscores, descriptive (e.g., `vector_add`, `matrix_multiplication`, `relu_activation`)

---

## Challenge Types & Difficulty Levels

### Easy Challenges

**Definition**: Single core concept, basic kernel launches, simple parallelization strategy.

**Characteristics**:
- 1-2 input parameters plus output
- Element-wise operations or basic matrix operations
- Clear, straightforward algorithmic approach
- Minimal optimization required
- Starter code includes thread/block calculation hints
- Broader coverage of fundamental GPU concepts

**Examples**:
- Vector addition/subtraction
- Matrix transposition
- Element-wise operations (ReLU, sigmoid, etc.)
- Simple array indexing operations

**Starter Code Approach**: Provide significant scaffolding including:
- Basic kernel function signature (with `__global__` in CUDA)
- Thread/block grid calculation in `solve()` 
- Kernel launch setup
- Synchronization calls

### Medium Challenges

**Definition**: Multiple concepts, memory optimizations, shared memory usage, more complex patterns.

**Characteristics**:
- 2-4 input/output parameters
- Requires understanding of memory hierarchies (global, shared, local)
- May involve reduction patterns, tiling strategies
- Needs optimization thinking (memory coalescing, bank conflicts)
- Starter code provides minimal structure
- Tests cover correctness and some performance aspects

**Examples**:
- Matrix multiplication with tiling
- 2D convolution
- Scan/prefix sum operations
- Histogram computation

**Starter Code Approach**: Minimal scaffolding:
- Empty `solve()` function with correct function signature
- Let users design grid/block strategy
- Optional: Empty kernel function (signature only)

### Hard Challenges

**Definition**: Advanced techniques, complex algorithms, significant optimization requirements.

**Characteristics**:
- Multiple input/output parameters with complex relationships
- Advanced optimization techniques (warp-level operations, cooperative groups, etc.)
- Non-trivial algorithms (sorting, graph algorithms, etc.)
- Heavy performance expectations
- Minimal starter guidance
- Strict performance benchmarks in tests

**Examples**:
- Optimized matrix multiplication with persistent kernels
- GPU-accelerated sorting (quick sort, merge sort)
- Graph algorithms on GPU
- Complex neural network operations

**Starter Code Approach**: Bare minimum:
- Only the function signature
- Users implement everything from scratch
- May include `// TODO` comments

---

## Challenge.py Specification

The `challenge.py` file contains the reference implementation, test case generation, and metadata. It must inherit from `ChallengeBase`.

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
            atol=1e-05,              # Absolute tolerance for float comparisons
            rtol=1e-05,              # Relative tolerance for float comparisons
            num_gpus=1,              # Number of GPUs required (usually 1)
            access_tier="free"       # "free" or "premium" (for future use)
        )
```

**Parameters**:
- **`name`**: Display name shown on the website (string, clear and descriptive)
- **`atol`**: Absolute tolerance for numerical comparisons in tests (float, typically 1e-5 or 1e-6)
- **`rtol`**: Relative tolerance for numerical comparisons (float, typically 1e-5)
- **`num_gpus`**: Number of GPU devices needed (int, usually 1)
- **`access_tier`**: Access level ("free" or "premium")

### Reference Implementation (`reference_impl`)

The reference implementation must:
1. Accept the same parameters as the user's solution
2. Perform the correct computation
3. Include assertions for parameter validation (shape, dtype, device)
4. Store results in output parameters (pass-by-reference)

```python
def reference_impl(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    # Validate inputs
    assert A.shape == B.shape == C.shape, "All tensors must have same shape"
    assert A.dtype == B.dtype == C.dtype, "All tensors must have same dtype"
    assert A.device == B.device == C.device, "All tensors must be on same device"
    
    # Perform computation
    torch.add(A, B, out=C)
```

**Key Points**:
- Use PyTorch operations for reference (works on GPU)
- Validate inputs to catch test case errors early
- Use `out=` parameter to modify tensors in-place where applicable
- Use `.copy_()` method when output is a separate tensor
- Handle both single and batch inputs appropriately

### Solve Signature (`get_solve_signature`)

This method defines the function signature that users must implement. It maps parameter names to ctypes specifications.

```python
def get_solve_signature(self) -> Dict[str, tuple]:
    return {
        "A": (ctypes.POINTER(ctypes.c_float), "in"),
        "B": (ctypes.POINTER(ctypes.c_float), "in"),
        "C": (ctypes.POINTER(ctypes.c_float), "out"),
        "N": (ctypes.c_size_t, "in"),
    }
```

**Format**: `"param_name": (ctype_spec, direction)`

**Common ctypes**:
- `ctypes.POINTER(ctypes.c_float)` - Float pointer (device memory)
- `ctypes.POINTER(ctypes.c_int)` - Int pointer (device memory)
- `ctypes.POINTER(ctypes.c_double)` - Double pointer (device memory)
- `ctypes.c_float` - Single float value
- `ctypes.c_int` - Integer value
- `ctypes.c_size_t` - Size value (for array lengths)
- `torch.Tensor` - PyTorch tensor (for frameworks that support it)
- `torch.nn.Module` - Neural network module

**Directions**:
- `"in"` - Input parameter
- `"out"` - Output parameter (written by the solution)

### Example Test Case (`generate_example_test`)

Generates a single simple test case for display on the challenge page.

```python
def generate_example_test(self) -> Dict[str, Any]:
    dtype = torch.float32
    N = 4
    A = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=dtype)
    B = torch.tensor([5.0, 6.0, 7.0, 8.0], device="cuda", dtype=dtype)
    C = torch.empty(N, device="cuda", dtype=dtype)
    
    return {
        "A": A,
        "B": B,
        "C": C,
        "N": N,
    }
```

**Guidelines**:
- Use small, simple values for clarity
- Make example output manually verifiable
- Should match or be similar to an example in `challenge.html`
- Include all parameters in the returned dictionary
- Always allocate output tensors with `torch.empty()`, not zeros
- Use `device="cuda"` consistently

### Functional Tests (`generate_functional_test`)

Generates 10-15 test cases covering:
- Edge cases (empty, single element, very small)
- Power-of-two and non-power-of-two sizes
- Various data ranges (zeros, negative, large numbers, very small numbers)
- Typical use cases
- Boundary conditions

```python
def generate_functional_test(self) -> List[Dict[str, Any]]:
    dtype = torch.float32
    test_cases = []
    
    # Named test specifications: (name, a_values, b_values)
    test_specs = [
        ("scalar_tail_1", [1.0], [2.0]),
        ("scalar_tail_2", [1.0, 2.0], [3.0, 4.0]),
        ("basic_small", [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]),
        ("all_zeros", [0.0] * 16, [0.0] * 16),
        ("non_power_of_two", [1.0] * 30, [2.0] * 30),
        ("negative_numbers", [-1.0, -2.0, -3.0], [-5.0, -6.0, -7.0]),
        ("mixed_positive_negative", [1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]),
        ("very_small_numbers", [1e-6, 1e-7, 1e-8], [1e-6, 1e-7, 1e-8]),
        ("large_numbers", [1e6, 1e7, -1e6], [1e6, -1e7, -1e6]),
    ]
    
    for _, a_vals, b_vals in test_specs:
        n = len(a_vals)
        test_cases.append({
            "A": torch.tensor(a_vals, device="cuda", dtype=dtype),
            "B": torch.tensor(b_vals, device="cuda", dtype=dtype),
            "C": torch.zeros(n, device="cuda", dtype=dtype),
            "N": n,
        })
    
    # Random test cases
    for size, a_range, b_range in [
        (32, (0.0, 32.0), (0.0, 64.0)),
        (1000, (0.0, 7.0), (0.0, 5.0)),
        (10000, (0.0, 1.0), (0.0, 1.0)),
    ]:
        test_cases.append({
            "A": torch.empty(size, device="cuda", dtype=dtype).uniform_(*a_range),
            "B": torch.empty(size, device="cuda", dtype=dtype).uniform_(*b_range),
            "C": torch.zeros(size, device="cuda", dtype=dtype),
            "N": size,
        })
    
    return test_cases
```

**Edge Cases to Cover**:
- **Size 1**: Single element (tests minimal parallelism)
- **Powers of 2**: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
- **Non-powers of 2**: 3, 5, 7, 30, 33, 63, 100, 255, 257
- **Zero values**: All zeros input and output
- **Negative numbers**: Pure negative, mixed
- **Very large/small numbers**: Test numerical stability
- **Boundary values**: 0, 1, -1, INT_MAX/2, FLT_MAX/1000

**Size Progression**:
- **Named tests**: 1-1024 elements (comprehensive coverage)
- **Random tests**: 32, 1000, 10000 elements (typical scales)
- Total functional tests: ~12-15 test cases

### Performance Test (`generate_performance_test`)

Generates a single large test case to benchmark performance.

```python
def generate_performance_test(self) -> Dict[str, Any]:
    dtype = torch.float32
    N = 25_000_000  # 25 million elements
    
    return {
        "A": torch.empty(N, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
        "B": torch.empty(N, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
        "C": torch.zeros(N, device="cuda", dtype=dtype),
        "N": N,
    }
```

**Guidelines**:
- Size should challenge GPU capabilities but still complete in seconds
- For 1D operations: 10M-100M elements
- For 2D operations: 4K×4K to 8K×8K matrices
- For complex operations: Scale down appropriately
- Realistic value ranges (not all zeros or ones)

**Typical Sizes by Operation**:
- **Vector operations**: 25-100M elements
- **Matrix operations (small)**: 4K×4K matrices
- **Matrix operations (large)**: 8K×8K matrices
- **Complex algorithms**: 1M-10M elements adjusted for complexity

---

## Challenge.html Specification

The HTML file presents the problem to users. It must be clean, clear, and well-formatted as an HTML fragment (no boilerplates, styles, or scripts).

### Required Sections

#### 1. Problem Description

```html
<p>
  Implement a program that performs element-wise addition of two vectors 
  containing 32-bit floating point numbers on a GPU.
  The program should take two input vectors of equal length and produce 
  a single output vector containing their sum.
</p>
```

**Guidelines**:
- 2-3 sentences maximum
- Clearly state what the function must do
- Mention data types (float32, int32, etc.)
- Specify constraints early

#### 2. Implementation Requirements

```html
<h2>Implementation Requirements</h2>
<ul>
  <li>External libraries are not permitted</li>
  <li>The <code>solve</code> function signature must remain unchanged</li>
  <li>The final result must be stored in vector <code>C</code></li>
</ul>
```

**Common Requirements**:
- "External libraries are not permitted" (unless algorithm requires specific libraries)
- "The solve function signature must remain unchanged"
- Output storage location and format
- Any memory constraints
- Performance expectations (for hard problems)

#### 3. Examples (1-3 minimum)

```html
<h2>Example 1:</h2>
<pre>
Input:  A = [1.0, 2.0, 3.0, 4.0]
        B = [5.0, 6.0, 7.0, 8.0]
Output: C = [6.0, 8.0, 10.0, 12.0]
</pre>

<h2>Example 2:</h2>
<pre>
Input:  A = [1.5, 1.5, 1.5]
        B = [2.3, 2.3, 2.3]
Output: C = [3.8, 3.8, 3.8]
</pre>
```

**Guidelines**:
- Provide 1-3 clear input/output examples
- Examples should be manually verifiable
- Show edge cases if relevant (e.g., single element)
- Match or include the example from `generate_example_test()`
- Use `<pre>` tags for input/output formatting

#### 4. Constraints

```html
<h2>Constraints</h2>
<ul>
  <li>Input vectors <code>A</code> and <code>B</code> have identical lengths</li>
  <li>1 ≤ <code>N</code> ≤ 100,000,000</li>
</ul>
```

**Guidelines**:
- Specify size constraints (min/max)
- Data type constraints
- Value ranges (if applicable)
- Special constraints (e.g., "matrices must be square")
- Use HTML entities for math symbols: `&le;` (≤), `&ge;` (≥), `&ne;` (≠)

#### 5. Optional: Time/Space Complexity (for medium/hard)

```html
<h2>Complexity Hints</h2>
<ul>
  <li>Expected time complexity: O(N) with proper parallelization</li>
  <li>Memory requirements: O(N) for inputs and outputs</li>
</ul>
```

### HTML Formatting Standards

```html
<!-- Code snippets in challenges -->
<code>variable_name</code>
<code>solve()</code>

<!-- Mathematical expressions -->
The time complexity is O(N).
Matrix dimensions: N &times; M (use &times; for multiplication)

<!-- Multiple line code -->
<pre>
code here
line by line
</pre>
```

---

## Starter Code Guidelines

Starter code files teach users the framework and problem structure. Different frameworks have different starter conventions.

### General Principles

1. **Compilation**: All starter code must compile/run without errors
2. **Non-functional**: Starters should NOT solve the problem (empty kernels, `pass` statements)
3. **Comments**: Follow the exact comment style and formatting of existing starter files
4. **Consistency**: Match the style of existing starters for that framework exactly

### By Framework

#### CUDA (`starter.cu`)

**Easy Challenge Structure**:
```cpp
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```

**Medium Challenge Structure**:
```cpp
#include <cuda_runtime.h>

__global__ void matrix_multiply(const float* A, const float* B, float* C, 
                                int N, int M, int K) {}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, 
                     int N, int M, int K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    matrix_multiply<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N, M, K);
    cudaDeviceSynchronize();
}
```

**Hard Challenge Structure**:
```cpp
#include <cuda_runtime.h>

extern "C" void solve(const float* A, const float* B, float* C, int N) {}
```

#### PyTorch (`starter.pytorch.py`)

**Easy Challenge Structure**:
```python
import torch


# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    pass
```

**Medium/Hard Challenge Structure**:
```python
import torch


# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    pass
```

#### Triton (`starter.triton.py`)

**Easy/Medium Structure**:
```python
import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pass


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)
```

#### Mojo (`starter.mojo`)

**Easy/Medium Structure**:
```mojo
from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn vector_add_kernel(A: UnsafePointer[Float32], B: UnsafePointer[Float32], C: UnsafePointer[Float32], N: Int32):
    pass

# A, B, C are device pointers (i.e. pointers to memory on the GPU)
@export
def solve(A: UnsafePointer[Float32], B: UnsafePointer[Float32], C: UnsafePointer[Float32], N: Int32):
    var BLOCK_SIZE: Int32 = 256
    var ctx = DeviceContext()
    var num_blocks = ceildiv(N, BLOCK_SIZE)

    ctx.enqueue_function[vector_add_kernel](
        A, B, C, N,
        grid_dim  = num_blocks,
        block_dim = BLOCK_SIZE
    )

    ctx.synchronize()
```

#### CuTe (`starter.cute.py`)

**Medium/Hard Structure**:
```python
import cutlass
import cutlass.cute as cute


# A, B, C are tensors on the GPU
@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    pass
```

#### JAX (`starter.jax.py`)

**Easy/Medium Structure**:
```python
import jax
import jax.numpy as jnp


# A, B are tensors on GPU
@jax.jit
def solve(A: jax.Array, B: jax.Array, N: int) -> jax.Array:
    # return output tensor directly
    pass
```

### Starter Code Comments Style

Comments in starter files follow a strict, minimal format:
- Only parameter descriptions (what is passed to the function)
- Parameter locations (on GPU, data type)
- No "TODO" comments or hints
- No explanations of the algorithm

**Example**:
```python
# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    pass
```

Not:
```python
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    """
    Add two vectors element-wise.
    Args:
        A: First input vector
        B: Second input vector
        C: Output vector
        N: Length of vectors
    """
    # TODO: Implement vector addition
    # Hint: Use torch operations
    pass
```

---

## Test Case Design

### Test Case Sizing Strategy

Test cases should progress from simple to complex to thoroughly validate correctness and performance.

#### Functional Test Progression

| Type | Size Range | Purpose | Count |
|------|-----------|---------|-------|
| Edge cases | 1-8 elements | Minimal parallelism, boundary conditions | 3-4 |
| Power-of-2 | 16-1024 elements | Common thread block configurations | 3-4 |
| Non-power-of-2 | 30, 33, 63, 100, 255, 257 | Irregular workload distribution | 3-4 |
| Random sizes | 1K-10K elements | Typical realistic sizes | 2-3 |
| **Total** | - | - | **12-15 cases** |

#### Performance Test Sizing

| Operation Type | Recommended Size | Rationale |
|---|---|---|
| 1D vector operations | 10M-100M elements | Saturate memory bandwidth |
| 2D matrix operations | 4K×4K to 8K×8K | Moderate complexity, completes in seconds |
| Complex algorithms | 1M-10M elements | Depends on algorithmic complexity |
| Neural network ops | Batch size 16-256 | Realistic workload |

### Numerical Stability in Tests

When working with floating-point operations:

```python
# Use appropriate tolerances
atol=1e-5   # For float32
atol=1e-6   # For higher precision requirements
rtol=1e-5   # Relative tolerance

# Avoid extreme value ranges that cause overflow/underflow
range = (-1000.0, 1000.0)  # Good
range = (-1e30, 1e30)      # Causes overflow
range = (-1e-30, 1e-30)    # Causes underflow
```

### Test Case Organization

Organize test cases by category:

```python
def generate_functional_test(self) -> List[Dict[str, Any]]:
    dtype = torch.float32
    test_cases = []
    
    # 1. Edge cases
    # ... single element, small arrays
    
    # 2. Power-of-two sizes
    # ... 16, 32, 64, 256, 1024
    
    # 3. Non-power-of-two sizes
    # ... 3, 5, 7, 30, 33, 63
    
    # 4. Special values
    # ... zeros, negatives, mixed
    
    # 5. Random sizes
    # ... realistic distributions
    
    return test_cases
```

### Data Type Considerations

```python
# Match the challenge's data type
dtype = torch.float32      # For float challenges
dtype = torch.float64      # For double precision
dtype = torch.int32        # For integer challenges

# Allocate output with correct dtype
C = torch.empty(N, device="cuda", dtype=dtype)
```

---

## Creating & Testing Challenges

### Manual Creation Process

#### Step 1: Create Directory Structure

```bash
mkdir -p challenges/easy/5_my_challenge/starter
```

#### Step 2: Write challenge.py

- Start with the template from an existing challenge
- Implement `reference_impl()` carefully
- Generate diverse test cases

```bash
# Test locally by importing:
python -c "from challenges.easy.5_my_challenge.challenge import Challenge; c = Challenge()"
```

#### Step 3: Write challenge.html

- Create clear problem description
- Include 1-3 examples
- Specify constraints precisely

#### Step 4: Generate Starter Code

Use the provided script:

```bash
python scripts/generate_starter_code.py challenges/easy/5_my_challenge
```

Or create them manually, following the framework guidelines.

#### Step 5: Validate & Test

```bash
# Run tests
python -m pytest tests/ -v

# Check formatting
pre-commit run --all-files
```

### Using generate_starter_code.py

The script automatically generates starter templates for all frameworks:

```bash
# For a specific challenge (absolute or relative path)
python scripts/generate_starter_code.py challenges/easy/5_my_challenge

# Script creates:
# - starter/starter.cu
# - starter/starter.pytorch.py
# - starter/starter.triton.py
# - starter/starter.mojo
# - starter/starter.cute.py
# - starter/starter.jax.py
```

**Generated starters**:
- Easy: Include grid/block setup hints
- Medium: Minimal starter structure
- Hard: Just function signatures

### Testing Challenges Locally

```python
# Test challenge.py file
from challenges.easy.5_my_challenge.challenge import Challenge

challenge = Challenge()

# Test example case
example = challenge.generate_example_test()
challenge.reference_impl(**example)

# Test a functional case
functionals = challenge.generate_functional_test()
for test_case in functionals:
    challenge.reference_impl(**test_case)

# Test performance case
perf_test = challenge.generate_performance_test()
challenge.reference_impl(**perf_test)
```

### Validating Test Case Coverage

Ensure your functional tests cover:

```
✓ Single element (N=1)
✓ Multiple edge cases (N=2,3,4)
✓ Powers of 2 up to 1024
✓ Non-powers of 2 (30, 33, 63, 100, 255, 257)
✓ All-zero inputs
✓ Negative numbers
✓ Mixed positive/negative
✓ Very large numbers
✓ Very small numbers
✓ Typical scales (1K, 10K, 100K elements)
```

---

## Formatting & Linting

All code must pass automated linting before merging. This is enforced by CI.

### Python Code (`.py` files)

**Tools**: black (formatting), isort (imports), flake8 (style)

```bash
# Install tools
pip install black==24.1.1 isort==5.13.2 flake8==7.0.0

# Format code
black challenges/ scripts/
isort challenges/ scripts/
flake8 challenges/ scripts/
```

**Key rules**:
- Line length: 100 characters max (black)
- 4 spaces for indentation
- Import order: stdlib → third-party → local
- Descriptive variable names

### CUDA/C++ Code (`.cu`, `.cpp`, `.h`)

**Tool**: clang-format

```bash
# Install on Ubuntu/Debian
sudo apt-get install clang-format

# Format code
find challenges -name "*.cu" -o -name "*.cpp" | xargs clang-format -i
```

**Style**: LLVM style with modifications

### Mojo Code (`.mojo`)

**Validation**: Basic checks (file not empty, imports, decorators)

- Follow Python-like conventions
- Use `@export` decorator for public functions
- Clear function signatures

### Automatic Pre-commit Hooks (Recommended)

```bash
# Install pre-commit
pip install pre-commit

# Install hooks in repo
pre-commit install

# Run manually
pre-commit run --all-files
```

This automatically formats code before each commit.

---

## Directory Structure Checklist

When adding a new challenge, verify:

```
✓ Directory name follows pattern: <number>_<name>
✓ challenge.html exists with:
  - Problem description (1-3 sentences)
  - Implementation requirements (list)
  - 1-3 examples (with input/output)
  - Constraints section

✓ challenge.py exists with:
  - Challenge class inheriting ChallengeBase
  - __init__ with proper metadata
  - reference_impl() with assertions
  - get_solve_signature() with ctypes
  - generate_example_test() (1 simple case)
  - generate_functional_test() (12-15 diverse cases)
  - generate_performance_test() (1 large case)

✓ starter/ directory contains:
  - starter.cu (CUDA)
  - starter.pytorch.py (PyTorch)
  - starter.triton.py (Triton)
  - starter.mojo (Mojo)
  - starter.cute.py (CuTe)
  - starter.jax.py (JAX)

✓ All code passes linting:
  - black, isort, flake8 for Python
  - clang-format for CUDA/C++

✓ Tests pass locally:
  - All functional tests execute without error
  - Performance test completes in <10 seconds
  - Reference implementation matches expected output
```

---

## Example: Creating a New Challenge Step-by-Step

### Challenge: Matrix Transpose (Easy)

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

**Step 4: Generate starters**
```bash
python scripts/generate_starter_code.py challenges/easy/3_matrix_transpose
```

**Step 5: Test locally**
```bash
python -m pytest tests/ -v
pre-commit run --all-files
```

---

## Common Pitfalls & Solutions

| Issue | Solution |
|-------|----------|
| Test cases all pass but performance is terrible | Increase performance test size; check if solution is doing unnecessary work |
| Inconsistent numerical results across frameworks | Ensure tolerance values (atol/rtol) match precision capabilities |
| Starter code doesn't compile | Test locally before submitting; check imports and syntax |
| Test sizes inconsistent between easy/medium/hard | Reference this guide's sizing recommendations |
| HTML formatting looks broken | Use proper HTML entities (&le;, &ge;, &times;) |
| Reference implementation is too slow | Optimize using PyTorch kernels rather than Python loops |

---

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

---

## Contributing Your Challenge

1. **Fork** the repository
2. **Create branch**: `git checkout -b challenge/your-challenge-name`
3. **Follow this guide** to create your challenge
4. **Run linting**: `pre-commit run --all-files`
5. **Run tests**: `python -m pytest tests/ -v`
6. **Commit & push** to your branch
7. **Submit PR** with clear description of the challenge

See [CONTRIBUTING.md](CONTRIBUTING.md) for additional guidelines and contributor terms.
