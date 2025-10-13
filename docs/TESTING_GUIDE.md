# Testing Guide for LeetGPU Challenges

This guide covers how to create test cases and validate your challenges to ensure they work correctly across all frameworks.

## Table of Contents

1. [Test Case Types](#test-case-types)
2. [Test Case Design Principles](#test-case-design-principles)
3. [Creating Robust Test Cases](#creating-robust-test-cases)
4. [Edge Cases and Boundary Conditions](#edge-cases-and-boundary-conditions)
5. [Performance Testing](#performance-testing)
6. [Validation Strategies](#validation-strategies)
7. [Common Testing Patterns](#common-testing-patterns)
8. [Debugging Test Issues](#debugging-test-issues)

## Test Case Types

### 1. Example Test (`generate_example_test`)
- **Purpose**: Simple test case that matches the example in `challenge.html`
- **Complexity**: Low - should be easy to understand and verify manually
- **Size**: Small (typically 3-10 elements)
- **Values**: Simple, predictable values

### 2. Functional Tests (`generate_functional_test`)
- **Purpose**: Comprehensive test suite covering various scenarios
- **Complexity**: Medium - includes edge cases and typical usage
- **Size**: Varied (small to medium)
- **Values**: Diverse, including edge cases

### 3. Performance Test (`generate_performance_test`)
- **Purpose**: Large test case for performance evaluation
- **Complexity**: High - tests scalability and efficiency
- **Size**: Large (typically 1M+ elements)
- **Values**: Random or structured large datasets

## Test Case Design Principles

### 1. Coverage
- **Input ranges**: Test minimum, maximum, and typical values
- **Input sizes**: Test small, medium, and large inputs
- **Data patterns**: Test edge cases, special values, and random data
- **Error conditions**: Test boundary conditions and invalid inputs

### 2. Determinism
- **Reproducible**: Tests should produce the same results every time
- **Seeded randomness**: Use fixed seeds for random test cases
- **Clear expectations**: Expected outputs should be well-defined

### 3. Efficiency
- **Fast execution**: Tests should run quickly for development
- **Memory efficient**: Avoid unnecessarily large test cases
- **Scalable**: Performance tests should be appropriately sized

## Debugging Test Issues

### Common Issues and Solutions

#### 1. Memory Issues
```python
# Problem: CUDA out of memory
# Solution: Reduce test case sizes
def generate_performance_test(self) -> Dict[str, Any]:
    # Reduce size if memory issues occur
    size = 100_000  # Instead of 1_000_000
    return {
        "input": torch.empty(size, device="cuda", dtype=torch.float32).uniform_(-100.0, 100.0),
        "output": torch.empty(size, device="cuda", dtype=torch.float32),
        "N": size
    }
```

#### 2. Precision Issues
```python
# Problem: Floating point precision errors
# Solution: Adjust tolerances
def __init__(self):
    super().__init__(
        name="Complex Algorithm",
        atol=1e-03,  # Increase tolerance for complex algorithms
        rtol=1e-03,
        num_gpus=1,
        access_tier="free"
    )
```

#### 3. Shape Mismatch Issues
```python
# Problem: Tensor shape mismatches
# Solution: Add shape validation
def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int):
    # Validate shapes
    assert input.shape == (N,), f"Expected input shape ({N},), got {input.shape}"
    assert output.shape == (N,), f"Expected output shape ({N},), got {output.shape}"
    
    # Rest of implementation...
```

### Debugging Checklist

- [ ] Reference implementation produces correct results
- [ ] All test cases have required parameters
- [ ] Tensor shapes match expectations
- [ ] Data types are consistent (float32)
- [ ] Tolerances are appropriate for the algorithm
- [ ] Performance test size is reasonable
- [ ] Edge cases are covered
- [ ] Random test cases use appropriate ranges

---

*This testing guide ensures your challenges are robust, well-tested, and ready for production use.* 