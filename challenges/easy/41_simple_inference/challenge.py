from typing import Any, List, Dict
import torch
import torch.nn as nn
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Simple Inference",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )
        
    def reference_impl(self, input: torch.Tensor, model: nn.Module, output: torch.Tensor):
        assert input.device == output.device
        assert input.dtype == output.dtype
        
        with torch.no_grad():
            result = model(input)
            output.copy_(result)
        
    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (torch.Tensor, "in"),
            "model": (nn.Module, "in"),
            "output": (torch.Tensor, "out")
        }
        
    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = "cuda"
        
        # Create a simple linear model
        model = nn.Linear(2, 2)
        model.weight.data = torch.tensor([[0.5, 1.0], [1.5, 0.5]], dtype=dtype)
        model.bias.data = torch.tensor([0.1, 0.2], dtype=dtype)
        model = model.to(device)
        
        input = torch.tensor([[1.0, 2.0]], device=device, dtype=dtype)
        output = torch.empty((1, 2), device=device, dtype=dtype)
        
        return {
            "input": input,
            "model": model,
            "output": output
        }
    
    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        device = "cuda"
        tests = []
        
        # Test 1: Basic 2->2 linear layer
        model1 = nn.Linear(2, 2)
        model1.weight.data = torch.tensor([[0.5, 1.0], [1.5, 0.5]], dtype=dtype)
        model1.bias.data = torch.tensor([0.1, 0.2], dtype=dtype)
        model1 = model1.to(device)
        
        tests.append({
            "input": torch.tensor([[1.0, 2.0]], device=device, dtype=dtype),
            "model": model1,
            "output": torch.empty((1, 2), device=device, dtype=dtype)
        })
        
        # Test 2: Single input/output
        model2 = nn.Linear(1, 1)
        model2.weight.data = torch.tensor([[2.0]], dtype=dtype)
        model2.bias.data = torch.tensor([1.0], dtype=dtype)
        model2 = model2.to(device)
        
        tests.append({
            "input": torch.tensor([[1.0], [2.0], [3.0]], device=device, dtype=dtype),
            "model": model2,
            "output": torch.empty((3, 1), device=device, dtype=dtype)
        })
        
        # Test 3: No bias
        model3 = nn.Linear(3, 2, bias=False)
        model3.weight.data = torch.tensor([[1.0, 0.0, -1.0], [0.5, 1.5, 0.0]], dtype=dtype)
        model3 = model3.to(device)
        
        tests.append({
            "input": torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, -1.0]], device=device, dtype=dtype),
            "model": model3,
            "output": torch.empty((2, 2), device=device, dtype=dtype)
        })
        
        # Test 4: Batch processing
        model4 = nn.Linear(4, 3)
        model4.weight.data = torch.randn((3, 4), dtype=dtype) * 0.5
        model4.bias.data = torch.randn(3, dtype=dtype) * 0.1
        model4 = model4.to(device)
        
        tests.append({
            "input": torch.randn((8, 4), device=device, dtype=dtype),
            "model": model4,
            "output": torch.empty((8, 3), device=device, dtype=dtype)
        })
        
        # Test 5: Larger model
        model5 = nn.Linear(10, 5)
        model5.weight.data = torch.randn((5, 10), dtype=dtype) * 0.3
        model5.bias.data = torch.randn(5, dtype=dtype) * 0.2
        model5 = model5.to(device)
        
        tests.append({
            "input": torch.randn((16, 10), device=device, dtype=dtype),
            "model": model5,
            "output": torch.empty((16, 5), device=device, dtype=dtype)
        })
        
        # Test 6: Zero weights
        model6 = nn.Linear(2, 2)
        model6.weight.data = torch.zeros((2, 2), dtype=dtype)
        model6.bias.data = torch.tensor([1.0, -1.0], dtype=dtype)
        model6 = model6.to(device)
        
        tests.append({
            "input": torch.tensor([[5.0, 10.0]], device=device, dtype=dtype),
            "model": model6,
            "output": torch.empty((1, 2), device=device, dtype=dtype)
        })
        
        # Test 7: Identity-like transformation
        model7 = nn.Linear(3, 3)
        model7.weight.data = torch.eye(3, dtype=dtype)
        model7.bias.data = torch.zeros(3, dtype=dtype)
        model7 = model7.to(device)
        
        tests.append({
            "input": torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype),
            "model": model7,
            "output": torch.empty((2, 3), device=device, dtype=dtype)
        })
        
        # Test 8: Single batch, many features
        model8 = nn.Linear(20, 1)
        model8.weight.data = torch.ones((1, 20), dtype=dtype) * 0.05  # Sum with scaling
        model8.bias.data = torch.tensor([0.0], dtype=dtype)
        model8 = model8.to(device)
        
        tests.append({
            "input": torch.randn((1, 20), device=device, dtype=dtype),
            "model": model8,
            "output": torch.empty((1, 1), device=device, dtype=dtype)
        })
        
        return tests
    
    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = "cuda"
        
        # Large model for performance testing
        model = nn.Linear(512, 256)
        model.weight.data = torch.randn((256, 512), dtype=dtype) * 0.1
        model.bias.data = torch.randn(256, dtype=dtype) * 0.05
        model = model.to(device)
        
        batch_size = 1000
        input = torch.randn((batch_size, 512), device=device, dtype=dtype)
        output = torch.empty((batch_size, 256), device=device, dtype=dtype)
        
        return {
            "input": input,
            "model": model,
            "output": output
        }
