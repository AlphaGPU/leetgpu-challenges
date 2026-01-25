from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ChallengeBase(ABC):
    def __init__(self, name: str, atol: float, rtol: float, num_gpus: int, access_tier: str):
        self.name = name
        self.atol = atol
        self.rtol = rtol
        self.num_gpus = num_gpus
        self.access_tier = access_tier

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
