import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Ordinary Least Squares",
            atol=1e-02,
            rtol=1e-02,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, X: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, n_samples: int, n_features: int):
        # Reshape tensors to their proper dimensions
        X_reshaped = X.view(n_samples, n_features)
        y_reshaped = y.view(n_samples)
        
        # Compute X^T * X
        XTX = torch.matmul(X_reshaped.t(), X_reshaped)
        
        # Compute X^T * y
        XTy = torch.matmul(X_reshaped.t(), y_reshaped)
        
        # Solve the system using Cholesky decomposition
        L = torch.linalg.cholesky(XTX)
        
        # Manual forward substitution for L * z = X^T * y
        z = torch.zeros_like(XTy)
        for i in range(n_features):
            z[i] = XTy[i]
            for j in range(i):
                z[i] = z[i] - L[i, j] * z[j]
            z[i] = z[i] / L[i, i]
        
        # Manual backward substitution for L^T * beta = z
        result = torch.zeros_like(z)
        for i in range(n_features - 1, -1, -1):
            result[i] = z[i]
            for j in range(i + 1, n_features):
                result[i] = result[i] - L[j, i] * result[j]
            result[i] = result[i] / L[i, i]
        
        # Copy to output tensor
        beta.copy_(result)

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
        n_samples, n_features = 5, 3
        X = torch.tensor([
            [-0.23, -0.23, 1.52],
            [0.77, -0.47, 1.58],
            [-0.14, 0.65, 0.5],
            [-1.91, -1.72, 0.24],
            [-0.46, -0.47, 0.54]
        ], dtype=dtype, device="cuda")
        y = torch.tensor([83.01, 93.4, 47.33, -62.22, 13.06], dtype=dtype, device="cuda")
        beta = torch.empty(n_features, dtype=dtype, device="cuda")
        return {"X": X.flatten(), "y": y, "beta": beta, "n_samples": n_samples, "n_features": n_features}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        device = "cuda"
        tests = []

        # Test 1: simple_1d
        tests.append({
            "X": torch.tensor([
                [0.24799999594688416],
                [-0.0689999982714653],
                [0.3240000009536743],
                [0.7620000243186951],
                [-0.11699999868869781]
            ], dtype=dtype, device=device),
            "y": torch.tensor([
                0.12200000137090683,
                -0.01899999938905239,
                0.17000000178813934,
                0.37599998712539673,
                -0.05299999937415123
            ], dtype=dtype, device=device),
            "beta": torch.zeros(1, dtype=dtype, device=device),
            "n_samples": 5,
            "n_features": 1
        })

        # Test 2: simple_2d
        tests.append({
            "X": torch.tensor([
                [0.1289999932050705, -0.45399999618530273],
                [-0.1889999955892563, -0.2669999897480011],
                [0.42899999022483826, -0.2070000022649765],
                [0.24899999797344208, 1.0049999952316284],
                [0.6309999823570251, -0.2199999988079071],
                [-0.17299999296665192, 0.2280000001192093]
            ], dtype=dtype, device=device),
            "y": torch.tensor([
                -0.40700000524520874,
                -0.3709999918937683,
                0.013000000268220901,
                1.128000020980835,
                0.11500000208616257,
                0.13500000536441803
            ], dtype=dtype, device=device),
            "beta": torch.zeros(2, dtype=dtype, device=device),
            "n_samples": 6,
            "n_features": 2
        })

        # Test 3: square_3x3
        tests.append({
            "X": torch.tensor([
                [0.125, 0.6579999923706055, 0.6230000257492065],
                [-0.8019999861717224, -0.23399999737739563, -0.8579999804496765],
                [0.9290000200271606, 0.04399999976158142, 0.4740000069141388]
            ], dtype=dtype, device=device),
            "y": torch.tensor([
                1.6610000133514404,
                -1.930999994277954,
                1.2170000076293945
            ], dtype=dtype, device=device),
            "beta": torch.zeros(3, dtype=dtype, device=device),
            "n_samples": 3,
            "n_features": 3
        })

        # Test 4: overdetermined_8x3
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
            ], dtype=dtype, device=device),
            "y": torch.tensor([
                -0.17499999701976776,
                -2.618000030517578,
                -0.07400000095367432,
                0.4269999861717224,
                -1.2580000162124634,
                -0.6259999871253967,
                -1.2640000581741333,
                -0.41600000858306885
            ], dtype=dtype, device=device),
            "beta": torch.zeros(3, dtype=dtype, device=device),
            "n_samples": 8,
            "n_features": 3
        })

        # Test 5: medium_10x5
        tests.append({
            "X": torch.tensor([
                [0.2919999957084656, 0.6159999966621399, 0.41100001335144043, -0.4000000059604645, 0.20600000023841858],
                [-0.08799999952316284, -0.03700000047683716, -0.28299999237060547, -0.04699999839067459, 0.42899999022483826],
                [-0.4309999942779541, 0.00800000037997961, 0.7829999923706055, -0.23499999940395355, -0.19599999487400055],
                [0.40799999237060547, 0.03799999877810478, -0.05000000074505806, 0.8119999766349792, -0.6679999828338623],
                [-0.06800000369548798, -0.23899999260902405, -0.796999990940094, -0.4339999854564667, -0.01600000075995922],
                [-0.7639999985694885, -0.06199999898672104, -0.13099999725818634, 0.49799999594688416, 0.1589999943971634],
                [-0.01899999938905239, -0.03400000184774399, -0.22100000083446503, -0.23999999463558197, 0.026000000536441803],
                [-0.4869999885559082, -0.7170000076293945, -0.18000000715255737, 0.22699999809265137, -0.40299999713897705],
                [-1.347000002861023, 0.25099998712539673, -0.0020000000949949026, -0.19599999487400055, -0.07800000160932541],
                [0.22499999403953552, 0.593999981880188, -0.16699999570846558, -0.057999998331069946, 0.9179999828338623]
            ], dtype=dtype, device=device),
            "y": torch.tensor([
                1.1009999513626099,
                0.4620000123977661,
                -0.007000000216066837,
                0.1420000046491623,
                -2.3970000743865967,
                0.7590000033378601,
                -0.796999990940094,
                -1.7799999713897705,
                -1.003000020980835,
                2.617000102996826
            ], dtype=dtype, device=device),
            "beta": torch.zeros(5, dtype=dtype, device=device),
            "n_samples": 10,
            "n_features": 5
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = "cuda"
        n_samples = 32
        n_features = 32
        X = torch.eye(n_samples, dtype=dtype, device=device)
        y = torch.ones(n_samples, dtype=dtype, device=device)
        beta = torch.zeros(n_features, dtype=dtype, device=device)
        return {
            "X": X.flatten(),  # flattened as in your other examples
            "y": y,
            "beta": beta,
            "n_samples": n_samples,
            "n_features": n_features
        }
