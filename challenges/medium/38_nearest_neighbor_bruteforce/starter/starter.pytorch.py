import torch

# points : (N, 3) float32 tensor on GPU
# indices: (N,)   int32  tensor on GPU (to be filled in-place)
def solve(points: torch.Tensor, indices: torch.Tensor, N: int):
    """
    PyTorch starter stub.  Replace the body with a CUDA extension,
    Triton call, or torch-scripted kernel.  Leave the signature unchanged.
    """
    pass  # TODO
