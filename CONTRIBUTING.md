# Contributing to LeetGPU

Thank you for contributing to LeetGPU! We welcome new challenges and framework support.

## What You Can Contribute

### 1. New Challenges
Add new GPU programming problems with starter templates across multiple frameworks.

### 2. New Framework Support
Add starter templates for new GPU programming frameworks to existing challenges.

### 3. Bug Fixes
Fix issues with existing starter templates or problem descriptions.

## Adding a New Challenge

Each challenge should include:

```
challenges/<difficulty>/<number>_<name>/
├── challenge.html          # Problem description
├── challenge.py           # Reference implementation and metadata
└── starter/              # Starter templates
    ├── starter.cu           # CUDA 
    ├── starter.mojo         # Mojo 
    ├── starter.pytorch.py   # PyTorch 
    ├── starter.tinygrad.py  # TinyGrad 
    └── starter.triton.py    # Triton 
```

### Requirements
- Clear problem description with 1 or more examples
- Starter templates should follow the format of existing starter templates
- Follow existing challenge patterns

### Starter Template Guidelines
- **Comments**: Match the format of existing challenges
- **Easy problems**: Provide more starter code and helpful structure
- **Medium/Hard problems**: Just have an empty solve function
- All templates should compile but not solve the problem

### Difficulty Levels
- **Easy**: Single concept, basic kernel launches
- **Medium**: Multiple concepts, memory optimizations
- **Hard**: Advanced techniques, complex algorithms

## Submission Process

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b challenge/new-challenge-name`
3. **Add your challenge** with all required files
5. **Submit a pull request** with a clear description

## Getting Help

- Open an issue for questions
- Check existing challenges for examples

Thank you for helping improve LeetGPU! 