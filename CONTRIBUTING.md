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

> [!NOTE]
Use generate_starter_code.py in scripts folder to help generate the starter code for medium and hard problems

```bash
python scripts/generate_starter_code.py path/to/challenge_dir # can be either absolute or relative path
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

## Contributor Terms

By submitting code, documentation, or any other content to this repository
(“Contribution”), you confirm that:

1. **Ownership**  
   You are the sole author of the Contribution, or have obtained all
   necessary rights and permissions to grant the licenses below.

2. **Dual License to the Project**  
   You grant AlphaGPU and its downstream users **two** non‑exclusive,
   worldwide, royalty‑free licenses to your Contribution:

   **a.** Under the repository’s public license,  
   Creative Commons Attribution‑NonCommercial‑NoDerivatives 4.0  
   (“CC BY‑NC‑ND”), for all public distribution.

   **b.** Under a commercial‑friendly license to AlphaGPU itself:  
   – **Grant:** AlphaGPU may reproduce, distribute, adapt, sublicense,  
     and create derivative works of your Contribution for any purpose,  
     including commercial use.  
   – **No further conditions:** AlphaGPU need not attribute further or  
     seek additional permission beyond this agreement.

3. **No Additional Restrictions**  
   You may **not** impose any further terms on your Contribution that
   conflict with the licenses granted above.

4. **Pull Request = Acceptance**  
   Submitting a Pull Request constitutes acceptance of these terms.

