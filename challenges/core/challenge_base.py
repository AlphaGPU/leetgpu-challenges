from abc import ABC, abstractmethod
from typing import Any, Dict, List


class RandTensor:
    """Uniform random input in [low, high)."""

    def __init__(self, shape, low=0.0, high=1.0, dtype="float32"):
        self.shape = tuple(shape)
        self.low = low
        self.high = high
        self.dtype = dtype


class RandnTensor:
    """Normal (Gaussian) random input."""

    def __init__(self, shape, mean=0.0, std=1.0, dtype="float32"):
        self.shape = tuple(shape)
        self.mean = mean
        self.std = std
        self.dtype = dtype


class RandIntTensor:
    """Uniform integer random input in [low, high)."""

    def __init__(self, shape, low, high, dtype="int32"):
        self.shape = tuple(shape)
        self.low = low
        self.high = high
        self.dtype = dtype


class FullTensor:
    """Constant-filled input (covers zeros / ones / full)."""

    def __init__(self, shape, value=0.0, dtype="float32"):
        self.shape = tuple(shape)
        self.value = value
        self.dtype = dtype


class OutTensor:
    """Output buffer (by shape): materialized empty where outputs are written in
    place (torch), omitted where they're returned functionally (jax)."""

    def __init__(self, shape, dtype="float32"):
        self.shape = tuple(shape)
        self.dtype = dtype


class ChallengeBase(ABC):
    name: str
    atol: float
    rtol: float
    num_gpus: int
    access_tier: str

    def __init__(self, device: str = "cuda"):
        self.device = device

    @abstractmethod
    def reference_impl(self, *args, **kwargs):
        """
        Reference solution implementation.
        """
        pass

    @abstractmethod
    def get_solve_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for solution.

        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        pass

    @abstractmethod
    def generate_example_test(self) -> List[Dict[str, Any]]:
        """
        Generate an example test case for this problem.

        Returns:
            Dictionary with test case parameters
        """
        pass

    @abstractmethod
    def generate_functional_test(self) -> List[Dict[str, Any]]:
        """
        Generate functional test cases for this problem.

        Returns:
            List of test case dictionaries
        """
        pass

    @abstractmethod
    def generate_performance_test(self) -> List[Dict[str, Any]]:
        """
        Generate a performance test case for this problem.

        Returns:
            Dictionary with test case parameters
        """
        pass
