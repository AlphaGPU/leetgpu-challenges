import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "K-Means Clustering"
    atol = 0.0001
    rtol = 0.0001
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        data_x: torch.Tensor,
        data_y: torch.Tensor,
        labels: torch.Tensor,
        initial_centroid_x: torch.Tensor,
        initial_centroid_y: torch.Tensor,
        final_centroid_x: torch.Tensor,
        final_centroid_y: torch.Tensor,
        sample_size: int,
        k: int,
        max_iterations: int,
    ):
        assert data_x.shape == (sample_size,)
        assert data_y.shape == (sample_size,)
        assert initial_centroid_x.shape == (k,)
        assert initial_centroid_y.shape == (k,)
        assert final_centroid_x.shape == (k,)
        assert final_centroid_y.shape == (k,)
        assert labels.shape == (sample_size,)
        final_centroid_x.copy_(initial_centroid_x)
        final_centroid_y.copy_(initial_centroid_y)
        for _ in range(max_iterations):
            expanded_x = data_x.view(-1, 1) - final_centroid_x.view(1, -1)
            expanded_y = data_y.view(-1, 1) - final_centroid_y.view(1, -1)
            distances = expanded_x**2 + expanded_y**2
            labels.copy_(torch.argmin(distances, dim=1))
            for i in range(k):
                mask = labels == i
                if mask.any():
                    final_centroid_x[i] = data_x[mask].mean()
                    final_centroid_y[i] = data_y[mask].mean()

    def reference_impl_jax(
        self, data_x, data_y, initial_centroid_x, initial_centroid_y, sample_size, k, max_iterations
    ):
        import jax
        import jax.numpy as jnp

        final_centroid_x = initial_centroid_x
        final_centroid_y = initial_centroid_y
        labels = jnp.zeros((sample_size,), dtype=jnp.int32)
        for _ in range(max_iterations):
            expanded_x = data_x.reshape(-1, 1) - final_centroid_x.reshape(1, -1)
            expanded_y = data_y.reshape(-1, 1) - final_centroid_y.reshape(1, -1)
            distances = expanded_x**2 + expanded_y**2
            labels = jnp.argmin(distances, axis=1).astype(jnp.int32)
            onehot = jax.nn.one_hot(labels, k, dtype=data_x.dtype)
            counts = onehot.sum(axis=0)
            sum_x = jnp.matmul(data_x, onehot, precision=jax.lax.Precision.HIGHEST)
            sum_y = jnp.matmul(data_y, onehot, precision=jax.lax.Precision.HIGHEST)
            safe_counts = jnp.where(counts == 0, 1, counts)
            mean_x = sum_x / safe_counts
            mean_y = sum_y / safe_counts
            final_centroid_x = jnp.where(counts > 0, mean_x, final_centroid_x)
            final_centroid_y = jnp.where(counts > 0, mean_y, final_centroid_y)
        return labels, final_centroid_x, final_centroid_y

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "data_x": (ctypes.POINTER(ctypes.c_float), "in"),
            "data_y": (ctypes.POINTER(ctypes.c_float), "in"),
            "labels": (ctypes.POINTER(ctypes.c_int), "out"),
            "initial_centroid_x": (ctypes.POINTER(ctypes.c_float), "in"),
            "initial_centroid_y": (ctypes.POINTER(ctypes.c_float), "in"),
            "final_centroid_x": (ctypes.POINTER(ctypes.c_float), "out"),
            "final_centroid_y": (ctypes.POINTER(ctypes.c_float), "out"),
            "sample_size": (ctypes.c_int, "in"),
            "k": (ctypes.c_int, "in"),
            "max_iterations": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        sample_size, k, max_iterations = 4, 2, 10
        data_x = torch.tensor([1.0, 2.0, 8.0, 9.0], device=self.device, dtype=dtype)
        data_y = torch.tensor([1.0, 2.0, 8.0, 9.0], device=self.device, dtype=dtype)
        labels = torch.empty(sample_size, device=self.device, dtype=torch.int32)
        initial_centroid_x = torch.tensor([1.0, 8.0], device=self.device, dtype=dtype)
        initial_centroid_y = torch.tensor([1.0, 8.0], device=self.device, dtype=dtype)
        final_centroid_x = torch.empty(k, device=self.device, dtype=dtype)
        final_centroid_y = torch.empty(k, device=self.device, dtype=dtype)
        return {
            "data_x": data_x,
            "data_y": data_y,
            "labels": labels,
            "initial_centroid_x": initial_centroid_x,
            "initial_centroid_y": initial_centroid_y,
            "final_centroid_x": final_centroid_x,
            "final_centroid_y": final_centroid_y,
            "sample_size": sample_size,
            "k": k,
            "max_iterations": max_iterations,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        test_cases = []
        # basic_clustering
        data_x = torch.tensor(
            [1.0, 1.5, 1.2, 1.3, 1.1, 5.0, 5.2, 5.1, 5.3, 5.4, 10.1, 10.2, 10.0, 10.3, 10.5],
            device=self.device,
            dtype=dtype,
        )
        data_y = torch.tensor(
            [1.0, 1.5, 1.2, 1.3, 1.1, 5.0, 5.2, 5.1, 5.3, 5.4, 10.1, 10.2, 10.0, 10.3, 10.5],
            device=self.device,
            dtype=dtype,
        )
        labels = torch.empty(15, device=self.device, dtype=torch.int32)
        initial_centroid_x = torch.tensor([3.4, 7.1, 8.5], device=self.device, dtype=dtype)
        initial_centroid_y = torch.tensor([3.4, 7.1, 8.5], device=self.device, dtype=dtype)
        final_centroid_x = torch.empty(3, device=self.device, dtype=dtype)
        final_centroid_y = torch.empty(3, device=self.device, dtype=dtype)
        test_cases.append(
            {
                "data_x": data_x,
                "data_y": data_y,
                "labels": labels,
                "initial_centroid_x": initial_centroid_x,
                "initial_centroid_y": initial_centroid_y,
                "final_centroid_x": final_centroid_x,
                "final_centroid_y": final_centroid_y,
                "sample_size": 15,
                "k": 3,
                "max_iterations": 20,
            }
        )
        # single_cluster
        data_x = torch.tensor(
            [1.0, 1.2, 1.1, 1.3, 1.5, 1.4, 1.6, 1.2, 1.3, 1.1], device=self.device, dtype=dtype
        )
        data_y = torch.tensor(
            [1.0, 1.2, 1.1, 1.3, 1.5, 1.4, 1.6, 1.2, 1.3, 1.1], device=self.device, dtype=dtype
        )
        labels = torch.empty(10, device=self.device, dtype=torch.int32)
        initial_centroid_x = torch.tensor([1.0, 5.0, 10.0], device=self.device, dtype=dtype)
        initial_centroid_y = torch.tensor([1.0, 5.0, 10.0], device=self.device, dtype=dtype)
        final_centroid_x = torch.empty(3, device=self.device, dtype=dtype)
        final_centroid_y = torch.empty(3, device=self.device, dtype=dtype)
        test_cases.append(
            {
                "data_x": data_x,
                "data_y": data_y,
                "labels": labels,
                "initial_centroid_x": initial_centroid_x,
                "initial_centroid_y": initial_centroid_y,
                "final_centroid_x": final_centroid_x,
                "final_centroid_y": final_centroid_y,
                "sample_size": 10,
                "k": 3,
                "max_iterations": 10,
            }
        )
        # empty_clusters
        data_x = torch.tensor(
            [
                1.0,
                1.5,
                1.2,
                1.3,
                1.1,
                1.4,
                1.6,
                1.2,
                1.7,
                1.3,
                10.0,
                10.5,
                10.2,
                10.3,
                10.1,
                10.4,
                10.6,
                10.2,
                10.7,
                10.3,
            ],
            device=self.device,
            dtype=dtype,
        )
        data_y = torch.tensor(
            [
                1.0,
                1.5,
                1.2,
                1.3,
                1.1,
                1.4,
                1.6,
                1.2,
                1.7,
                1.3,
                10.0,
                10.5,
                10.2,
                10.3,
                10.1,
                10.4,
                10.6,
                10.2,
                10.7,
                10.3,
            ],
            device=self.device,
            dtype=dtype,
        )
        labels = torch.empty(20, device=self.device, dtype=torch.int32)
        initial_centroid_x = torch.tensor([1.5, 5.0, 10.5], device=self.device, dtype=dtype)
        initial_centroid_y = torch.tensor([1.5, 5.0, 10.5], device=self.device, dtype=dtype)
        final_centroid_x = torch.empty(3, device=self.device, dtype=dtype)
        final_centroid_y = torch.empty(3, device=self.device, dtype=dtype)
        test_cases.append(
            {
                "data_x": data_x,
                "data_y": data_y,
                "labels": labels,
                "initial_centroid_x": initial_centroid_x,
                "initial_centroid_y": initial_centroid_y,
                "final_centroid_x": final_centroid_x,
                "final_centroid_y": final_centroid_y,
                "sample_size": 20,
                "k": 3,
                "max_iterations": 15,
            }
        )
        # max_iterations_limit
        data_x = torch.tensor(
            [1.0, 1.5, 1.2, 1.3, 1.1, 5.0, 5.2, 5.1, 5.3, 5.4, 10.1, 10.2, 10.0, 10.3, 10.5],
            device=self.device,
            dtype=dtype,
        )
        data_y = torch.tensor(
            [1.0, 1.5, 1.2, 1.3, 1.1, 5.0, 5.2, 5.1, 5.3, 5.4, 10.1, 10.2, 10.0, 10.3, 10.5],
            device=self.device,
            dtype=dtype,
        )
        labels = torch.empty(15, device=self.device, dtype=torch.int32)
        initial_centroid_x = torch.tensor([3.4, 7.1, 8.5], device=self.device, dtype=dtype)
        initial_centroid_y = torch.tensor([3.4, 7.1, 8.5], device=self.device, dtype=dtype)
        final_centroid_x = torch.empty(3, device=self.device, dtype=dtype)
        final_centroid_y = torch.empty(3, device=self.device, dtype=dtype)
        test_cases.append(
            {
                "data_x": data_x,
                "data_y": data_y,
                "labels": labels,
                "initial_centroid_x": initial_centroid_x,
                "initial_centroid_y": initial_centroid_y,
                "final_centroid_x": final_centroid_x,
                "final_centroid_y": final_centroid_y,
                "sample_size": 15,
                "k": 3,
                "max_iterations": 5,
            }
        )
        # medium_random
        sample_size = 100
        k = 5
        data_x = torch.empty(sample_size, device=self.device, dtype=dtype).uniform_(0.0, 100.0)
        data_y = torch.empty(sample_size, device=self.device, dtype=dtype).uniform_(0.0, 100.0)
        labels = torch.empty(sample_size, device=self.device, dtype=torch.int32)
        initial_centroid_x = torch.tensor(
            [20.0, 40.0, 60.0, 80.0, 10.0], device=self.device, dtype=dtype
        )
        initial_centroid_y = torch.tensor(
            [20.0, 40.0, 60.0, 80.0, 50.0], device=self.device, dtype=dtype
        )
        final_centroid_x = torch.empty(k, device=self.device, dtype=dtype)
        final_centroid_y = torch.empty(k, device=self.device, dtype=dtype)
        test_cases.append(
            {
                "data_x": data_x,
                "data_y": data_y,
                "labels": labels,
                "initial_centroid_x": initial_centroid_x,
                "initial_centroid_y": initial_centroid_y,
                "final_centroid_x": final_centroid_x,
                "final_centroid_y": final_centroid_y,
                "sample_size": sample_size,
                "k": k,
                "max_iterations": 30,
            }
        )
        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        sample_size = 10000
        k = 5
        data_x = torch.empty(sample_size, device=self.device, dtype=dtype).uniform_(0.0, 1000.0)
        data_y = torch.empty(sample_size, device=self.device, dtype=dtype).uniform_(0.0, 1000.0)
        labels = torch.empty(sample_size, device=self.device, dtype=torch.int32)
        initial_centroid_x = torch.tensor(
            [100.0, 200.0, 300.0, 400.0, 500.0], device=self.device, dtype=dtype
        )
        initial_centroid_y = torch.tensor(
            [100.0, 200.0, 300.0, 400.0, 500.0], device=self.device, dtype=dtype
        )
        final_centroid_x = torch.empty(k, device=self.device, dtype=dtype)
        final_centroid_y = torch.empty(k, device=self.device, dtype=dtype)
        return {
            "data_x": data_x,
            "data_y": data_y,
            "labels": labels,
            "initial_centroid_x": initial_centroid_x,
            "initial_centroid_y": initial_centroid_y,
            "final_centroid_x": final_centroid_x,
            "final_centroid_y": final_centroid_y,
            "sample_size": sample_size,
            "k": k,
            "max_iterations": 30,
        }
