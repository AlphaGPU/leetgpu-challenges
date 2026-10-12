from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ChallengeBase(ABC):
    # Challenges declare these as class attributes. They may also be passed to
    # __init__ for backwards compatibility with older challenges.
    name: str
    atol: float
    rtol: float
    num_gpus: int = 1
    access_tier: str = "free"

    def __init__(
        self,
        name: Optional[str] = None,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        num_gpus: Optional[int] = None,
        access_tier: Optional[str] = None,
        device: str = "cuda",
    ):
        if name is not None:
            self.name = name
        if atol is not None:
            self.atol = atol
        if rtol is not None:
            self.rtol = rtol
        if num_gpus is not None:
            self.num_gpus = num_gpus
        if access_tier is not None:
            self.access_tier = access_tier
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
