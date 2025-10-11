import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Logistic Regression",
            atol=1e-02,
            rtol=1e-02,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, X: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, n_samples: int, n_features: int):
        """
        Logistic regression using Newton-Raphson (IRLS) in PyTorch.
        This converges faster and more accurately than plain gradient descent.
        """
        assert X.dtype == torch.float32
        assert y.dtype == torch.float32
        assert beta.dtype == torch.float32
        assert X.shape == (n_samples, n_features)
        assert y.shape == (n_samples,)
        assert beta.shape == (n_features,)
        
        X_reshaped = X.view(n_samples, n_features)
        y_reshaped = y.view(n_samples)
        beta.zero_()

        max_iter = 1000
        tol = 1e-8
        l2_reg = 1e-6

        for iteration in range(max_iter):
            z = torch.mv(X_reshaped, beta)
            p = torch.sigmoid(z)
            W = p * (1 - p)
            W = torch.clamp(W, min=1e-8)

            # Gradient
            gradient = torch.mv(X_reshaped.t(), p - y_reshaped) + l2_reg * beta

            # Hessian
            XW = X_reshaped * W.unsqueeze(1)
            hessian = torch.mm(X_reshaped.t(), XW) + l2_reg * torch.eye(n_features, device=X.device, dtype=X.dtype)

            # Solve H @ delta = gradient
            try:
                delta = torch.linalg.solve(hessian, gradient)
            except RuntimeError:
                delta = torch.linalg.lstsq(hessian, gradient.unsqueeze(1)).solution.squeeze()

            beta_new = beta - delta

            if torch.norm(beta_new - beta) < tol:
                beta.copy_(beta_new)
                break

            beta.copy_(beta_new)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "X": (ctypes.POINTER(ctypes.c_float), "in"),
            "y": (ctypes.POINTER(ctypes.c_float), "in"),
            "beta": (ctypes.POINTER(ctypes.c_float), "out"),
            "n_samples": (ctypes.c_int, "in"),
            "n_features": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        X = torch.tensor([
            [2.0, 1.0],
            [1.0, 2.0],
            [3.0, 3.0],
            [1.5, 2.5],
            [-1.0, -2.0],
            [-2.0, -1.0],
            [-1.5, -2.5],
            [-3.0, -3.0]
        ], device="cuda", dtype=dtype)
        y = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device="cuda", dtype=dtype)
        beta = torch.zeros(2, device="cuda", dtype=dtype)
        return {
            "X": X,
            "y": y,
            "beta": beta,
            "n_samples": 8,
            "n_features": 2
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # simple_1d
        tests.append({
            "X": torch.tensor([
                [0.24799999594688416],
                [-0.0689999982714653],
                [0.3240000009536743],
                [0.7620000243186951],
                [-0.11699999868869781]
            ], device="cuda", dtype=dtype),
            "y": torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0], device="cuda", dtype=dtype),
            "beta": torch.zeros(1, device="cuda", dtype=dtype),
            "n_samples": 5,
            "n_features": 1
        })

        # simple_2d
        tests.append({
            "X": torch.tensor([
                [0.1289999932050705, -0.45399999618530273],
                [-0.1889999955892563, -0.2669999897480011],
                [0.42899999022483826, -0.2070000022649765],
                [0.24899999797344208, 1.0049999952316284],
                [0.6309999823570251, -0.2199999988079071],
                [-0.17299999296665192, 0.2280000001192093]
            ], device="cuda", dtype=dtype),
            "y": torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device="cuda", dtype=dtype),
            "beta": torch.zeros(2, device="cuda", dtype=dtype),
            "n_samples": 6,
            "n_features": 2
        })

        # square_3x3
        tests.append({
            "X": torch.tensor([
                [0.125, 0.6579999923706055, 0.6230000257492065],
                [-0.8019999861717224, -0.23399999737739563, -0.8579999804496765],
                [0.9290000200271606, 0.04399999976158142, 0.4740000069141388]
            ], device="cuda", dtype=dtype),
            "y": torch.tensor([1.0, 0.0, 1.0], device="cuda", dtype=dtype),
            "beta": torch.zeros(3, device="cuda", dtype=dtype),
            "n_samples": 3,
            "n_features": 3
        })

        # overdetermined_8x3
        tests.append({
            "X": torch.tensor([
                [0.013000000268220901, 0.12999999523162842, -0.1979999989271164],
                [-0.10199999809265137, -0.6359999775886536, -1.2979999780654907],
                [0.14499999582767487, -0.43700000643730164, 0.19699999690055847],
                [0.46799999475479126, -0.00800000037997961, 0.12999999523162842],
                [-0.7369999885559082, 0.4009999930858612, -0.875],
                [-0.24799999594688416, -0.5040000081062317, 0.013000000268220901],
                [-0.061000000685453415, -0.7730000019073486, -0.30300000309944153],
                [-0.6970000267028809, -0.3140000104904175, 0.16599999368190765]
            ], device="cuda", dtype=dtype),
            "y": torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], device="cuda", dtype=dtype),
            "beta": torch.zeros(3, device="cuda", dtype=dtype),
            "n_samples": 8,
            "n_features": 3
        })

        # medium_10x3
        tests.append({
            "X": torch.tensor([
                [0.2919999957084656, 0.6159999966621399, 0.41100001335144043],
                [-0.4000000059604645, 0.20600000023841858, -0.08799999952316284],
                [-0.03700000047683716, -0.28299999237060547, -0.04699999839067459],
                [0.42899999022483826, -0.4309999942779541, 0.00800000037997961],
                [0.7829999923706055, -0.23499999940395355, -0.19599999487400055],
                [0.40799999237060547, 0.03799999877810478, -0.05000000074505806],
                [0.8119999766349792, -0.6679999828338623, -0.06800000369548798],
                [-0.23899999260902405, -0.796999990940094, -0.4339999854564667],
                [-0.01600000075995922, -0.7639999985694885, -0.06199999898672104],
                [-0.13099999725818634, 0.49799999594688416, 0.1589999943971634]
            ], device="cuda", dtype=dtype),
            "y": torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], device="cuda", dtype=dtype),
            "beta": torch.zeros(3, device="cuda", dtype=dtype),
            "n_samples": 10,
            "n_features": 3
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = "cuda"

        X = torch.eye(8, device=device, dtype=dtype).repeat(2, 1)
        y = torch.tensor([
            0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,
            0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0
        ], device=device, dtype=dtype)
        beta = torch.zeros(8, device=device, dtype=dtype)

        return {
            "X": X,
            "y": y,
            "beta": beta,
            "n_samples": 16,
            "n_features": 8
        }
