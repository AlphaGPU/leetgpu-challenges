# LeetGPU Challenge Creation Guide

## Directory Structure

```
challenges/<difficulty>/<number>_<name>/
├── challenge.html        # Problem description
├── challenge.py          # Reference impl, test cases, metadata
└── starter/              # One per framework
    ├── starter.cu
    ├── starter.cute.py
    ├── starter.jax.py
    ├── starter.mojo
    ├── starter.pytorch.py
    └── starter.triton.py
```

- **Naming**: `<number>_<challenge_name>` — sequential integer, lowercase with underscores
- **Linting & contribution process**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Starter code generation**: `python scripts/generate_starter_code.py path/to/challenge_dir`

## Difficulty Levels

| Level | Parameters | Concepts | Examples |
|-------|-----------|----------|----------|
| Easy | 1-2 in + output | Single concept, basic parallelization | Vector add, transpose, element-wise ops |
| Medium | 2-4 in/out | Memory hierarchies, reductions, tiling | Tiled matmul, 2D convolution |
| Hard | Multiple with complex relationships | Warp ops, cooperative groups, heavy perf | GPU sorting, graph algorithms |

## challenge.py

Must inherit from `ChallengeBase` and follow Black formatting (line length 100).

**Reference files to read for patterns:**
- Base class: `challenges/core/challenge_base.py`
- Simple example: `challenges/easy/1_vector_add/challenge.py`
- Matrix example: `challenges/easy/3_matrix_transpose/challenge.py`
- Medium example: `challenges/medium/22_gemm/challenge.py`

### Required Methods

#### `__init__`
```python
super().__init__(
    name="Challenge Display Name",
    atol=1e-05,           # Absolute tolerance (float32 default)
    rtol=1e-05,           # Relative tolerance (float32 default)
    num_gpus=1,
    access_tier="free"    # "free" or "premium"
)
```

#### `reference_impl(self, ...)`
- Same parameters as user's `solve` function
- Must include assertions on shape, dtype (`torch.float32`), and device (`cuda`)
- Use PyTorch operations (not Python loops) for performance

#### `get_solve_signature(self) -> Dict[str, tuple]`
Maps parameter names to `(ctype, direction)` tuples.

| ctypes | Use for |
|--------|---------|
| `ctypes.POINTER(ctypes.c_float)` | Tensor data |
| `ctypes.c_size_t` | Sizes/dimensions |
| `ctypes.c_int` | Integer parameters |

| Direction | Meaning |
|-----------|---------|
| `"in"` | Read-only input |
| `"out"` | Write-only output |
| `"inout"` | Read and write |

#### `generate_example_test(self) -> Dict[str, Any]`
One small, human-readable test case for display. Use literal tensor values.

#### `generate_functional_test(self) -> List[Dict[str, Any]]`
12-15 test cases with this coverage:

| Category | Sizes | Count |
|----------|-------|-------|
| Edge cases | 1, 2, 3, 4 | 3-4 |
| Power-of-2 | 16, 32, 64, 128, 256, 512, 1024 | 3-4 |
| Non-power-of-2 | 30, 100, 255 | 3-4 |
| Realistic | 1K-10K | 2-3 |

Must also include: zero inputs, negative numbers, mixed values.

#### `generate_performance_test(self) -> Dict[str, Any]`
One large test case. Size must fit 5x within 16GB (Tesla T4 VRAM).

| Operation type | Size |
|---------------|------|
| 1D | 10M-100M elements |
| 2D | 4K×4K to 8K×8K |
| Complex | 1M-10M |

## challenge.html

HTML fragment with four required sections:

1. **Problem description** — 2-3 sentences: what the function does, data types, constraints
2. **Implementation requirements** — Signature unchanged, no external libs, output location
3. **Examples** — 1-3 examples in `<pre>` blocks with Input/Output
4. **Constraints** — Size bounds, data types, value ranges, **and performance test size**

**Formatting rules:**
- `<code>` for variables/functions, `<pre>` for examples
- `&le;`, `&ge;`, `&times;` for math symbols
- **Performance test size bullet**: Must include a bullet documenting the exact parameters used in `generate_performance_test()`, formatted as:
  - `<li>Performance is measured with <code>param</code> = value</li>`
  - Use commas for numbers ≥ 1,000 (e.g., `25,000,000`)
  - Multiple parameters: `<code>M</code> = 8,192, <code>N</code> = 6,144, <code>K</code> = 4,096`

**Reference**: `challenges/easy/2_matrix_multiplication/challenge.html`

## Starter Code

Must compile/run without errors but not solve the problem. No comments except the parameter description comment (e.g., `// A, B, C are device pointers`).

**Rules:**
- Easy problems: provide kernel scaffold with grid/block setup
- Medium/Hard problems: empty `solve` function only
- Match the exact style of existing starters in each framework

**Reference files** (read these for exact format):
- CUDA: `challenges/easy/1_vector_add/starter/starter.cu`
- PyTorch: `challenges/easy/1_vector_add/starter/starter.pytorch.py`
- Triton: `challenges/easy/1_vector_add/starter/starter.triton.py`
- JAX: `challenges/easy/1_vector_add/starter/starter.jax.py`
- CuTe: `challenges/easy/1_vector_add/starter/starter.cute.py`
- Mojo: `challenges/easy/1_vector_add/starter/starter.mojo`

## Creation Workflow

1. Create directory: `mkdir -p challenges/<difficulty>/<number>_<name>/starter`
2. Write `challenge.py` — inherit ChallengeBase, implement all 5 methods
3. Write `challenge.html` — all 4 sections
4. Write starter code for all 6 frameworks (or use `scripts/generate_starter_code.py`)
5. Validate:
   ```bash
   python -c "from challenges.<difficulty>.<number>_<name>.challenge import Challenge; c = Challenge(); print('Tests:', len(c.generate_functional_test()))"
   ```
6. Lint: `pre-commit run --all-files`

## Testing with `run_challenge.py`

Use `scripts/run_challenge.py` to submit solutions against the live platform when creating or reviewing challenges. This reads `challenge.py` from the challenge directory and sends it along with the solution.

```bash
python scripts/run_challenge.py path/to/challenge_dir --language cuda --action run
```

**Rules:**
- **GPU**: Always use `--gpu "NVIDIA TESLA T4"` (the default). Do not use any other GPU.
- **Submission limit**: You may only run this script **5 times per session**. Use submissions carefully — verify your challenge locally (imports, assertions, lint) before submitting.
- **Workflow**: Write a CUDA solution in `solution/solution.cu`, run the script with `--action run` to validate, and only use `--action submit` when confident. Do not commit the solution file to the PR.

## Checklist

- [ ] Directory follows `<number>_<name>` convention
- [ ] `challenge.py`: inherits ChallengeBase, has reference_impl with assertions, all test generators
- [ ] `challenge.html`: description, requirements, examples, constraints with performance test size bullet
- [ ] `starter/`: all 6 framework files present, compilable, non-functional
- [ ] Functional tests: 12-15 cases covering edges, powers-of-2, non-powers, special values
- [ ] Performance test: appropriately sized for 16GB VRAM
- [ ] Linting passes: black, isort, flake8 (Python); clang-format (CUDA)
