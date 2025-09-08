# Creating New Challenges for LeetGPU

LeetGPU challenges are low-level GPU programming tasks focused on writing custom CUDA, Triton, or Tinygrad kernels. They evaluate both functional correctness and performance under real GPU constraints.

This guide provides instructions for creating new GPU programming challenges for LeetGPU. It covers the complete process from concept to submission.

## Challenge Structure

Each challenge follows this directory structure:

```
challenges/<difficulty>/<number>_<name>/
├── challenge.html          # Problem description and examples
├── challenge.py           # Reference implementation and test cases
└── starter/              # Starter templates for each framework
    ├── starter.cu           # CUDA template
    ├── starter.mojo         # Mojo template
    ├── starter.pytorch.py   # PyTorch template
    ├── starter.tinygrad.py  # TinyGrad template
    └── starter.triton.py    # Triton template
```

### Challenge.html template


# [Challenge Name]

## Description

[Provide a clear, concise explanation of what the algorithm or function is supposed to do. Include input and output specifications, if necessary.]

### Mathematical Formulation

[If applicable, provide the mathematical formula using LaTeX notation]

$$
\text{[Your formula here]}
$$

## Implementation Requirements

- **No External Libraries:** Solutions must be implemented using only native features. No external libraries or frameworks are permitted.
- **Function Signature:** The solve function signature is fixed and must not be modified. Implement your solution according to the provided signature.
- **Output Variable:** Results must be written to the designated output parameter: `[output_parameter_name]`



## Examples

### Example 1
**Input:**
```
[Provide specific input values]
```

**Expected Output:**
```
[Show the corresponding output values]
```

### Example 2
**Input:**
```
[Provide different input values]
```

**Expected Output:**
```
[Show the corresponding output values]
```

## Constraints

- **Input Size:** [Specify the range of input dimensions, e.g., "1 ≤ N ≤ 1,000,000"]
- **Value Range:** [Specify the range of input values, e.g., "-1000.0 ≤ input[i] ≤ 1000.0"]
- **Memory Limits:** [If applicable, specify any memory constraints]


