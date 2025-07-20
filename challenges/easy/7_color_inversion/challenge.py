import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Color Inversion",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )
        
    def reference_impl(self, image: torch.Tensor, width: int, height: int):
        assert image.shape == (height * width * 4,)
        assert image.dtype == torch.uint8
        
        # Reshape to (height, width, 4) for easier processing
        image_reshaped = image.view(height, width, 4)
        
        # Invert RGB channels (first 3 channels), keep alpha unchanged
        image_reshaped[:, :, :3] = 255 - image_reshaped[:, :, :3]
        
    def get_solve_signature(self) -> Dict[str, Any]:
        return {
            "image": ctypes.POINTER(ctypes.c_ubyte),
            "width": ctypes.c_int,
            "height": ctypes.c_int
        }
        
    def generate_example_test(self) -> Dict[str, Any]:
        width, height = 1, 2
        image = torch.tensor([255, 0, 128, 255, 0, 255, 0, 255], device="cuda", dtype=torch.uint8)
        return {
            "image": image,
            "width": width,
            "height": height
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "basic_2x2",
                "parameters": {
                    "image": {
                        "shape": [2, 2, 4],
                        "dtype": "uint8",
                        "value": [
                            [[255, 0, 0, 255], [0, 255, 0, 255]],
                            [[0, 0, 255, 255], [128, 128, 128, 255]]
                        ]
                    },
                    "width": {
                        "shape": [],
                        "dtype": "int32",
                        "value": 2
                    },
                    "height": {
                        "shape": [],
                        "dtype": "int32",
                        "value": 2
                    }
                }
            },
            {
                "id": "single_pixel",
                "parameters": {
                    "image": {
                        "shape": [1, 1, 4],
                        "dtype": "uint8",
                        "value": [
                            [[100, 50, 200, 255]]
                        ]
                    },
                    "width": {
                        "shape": [],
                        "dtype": "int32",
                        "value": 1
                    },
                    "height": {
                        "shape": [],
                        "dtype": "int32",
                        "value": 1
                    }
                }
            },
            {
                "id": "all_black",
                "parameters": {
                    "image": {
                        "shape": [3, 4, 4],
                        "dtype": "uint8",
                        "range": {"min": 0, "max": 0}
                    },
                    "width": {
                        "shape": [],
                        "dtype": "int32",
                        "value": 4
                    },
                    "height": {
                        "shape": [],
                        "dtype": "int32",
                        "value": 3
                    }
                }
            },
            {
                "id": "all_white",
                "parameters": {
                    "image": {
                        "shape": [5, 3, 4],
                        "dtype": "uint8",
                        "range": {"min": 255, "max": 255}
                    },
                    "width": {
                        "shape": [],
                        "dtype": "int32",
                        "value": 3
                    },
                    "height": {
                        "shape": [],
                        "dtype": "int32",
                        "value": 5
                    }
                }
            },
            {
                "id": "varying_alpha",
                "parameters": {
                    "image": {
                        "shape": [2, 2, 4],
                        "dtype": "uint8",
                        "value": [
                            [[10, 20, 30, 50], [40, 50, 60, 100]],
                            [[70, 80, 90, 150], [100, 110, 120, 200]]
                        ]
                    },
                    "image_output": {
                        "shape": [2, 2, 4],
                        "dtype": "uint8",
                        "range": {"min": 0, "max": 0}
                    },
                    "width": {
                        "shape": [],
                        "dtype": "int32",
                        "value": 2
                    },
                    "height": {
                        "shape": [],
                        "dtype": "int32",
                        "value": 2
                    }
                }
            },
            {
                "id": "large_square",
                "parameters": {
                    "image": {
                        "shape": [64, 64, 4],
                        "dtype": "uint8",
                        "range": {"min": 0, "max": 255}
                    },
                    "width": {
                        "shape": [],
                        "dtype": "int32",
                        "value": 64
                    },
                    "height": {
                        "shape": [],
                        "dtype": "int32",
                        "value": 64
                    }
                }
            },
            {
                "id": "rectangular",
                "parameters": {
                    "image": {
                        "shape": [32, 64, 4],
                        "dtype": "uint8",
                        "range": {"min": 0, "max": 255}
                    },
                    "width": {
                        "shape": [],
                        "dtype": "int32",
                        "value": 64
                    },
                    "height": {
                        "shape": [],
                        "dtype": "int32",
                        "value": 32
                    }
                }
            }
        ]

    def generate_performance_test(self) -> Dict[str, Any]:
        width, height = 4096, 5120
        size = width * height * 4
        return {
            "image": torch.randint(0, 256, (size,), device="cuda", dtype=torch.uint8),
            "width": width,
            "height": height
        }
