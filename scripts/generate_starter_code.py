import os
import sys
from importlib import util, machinery
import types
import ctypes
import urllib.request
import tempfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CONST_HINTS = {
    "input"
}

CTYPE_TO_CUDA = {
    ctypes.c_int: "int",
    ctypes.c_float: "float",
    ctypes.c_double: "double",
    ctypes.c_uint32: "unsigned int",
    ctypes.c_int64: "long long",
    ctypes.c_uint16: "__half",
}

CTYPE_TO_MOJO = {
    ctypes.c_int: "Int32",
    ctypes.c_float: "Float32",
    ctypes.c_double: "Float64",
    ctypes.c_uint32: "UInt32",
    ctypes.c_int64: "Int64",
    ctypes.c_uint16: "Float16",
}

CTYPE_TO_TORCH = {
    ctypes.c_int: "int",
    ctypes.c_float: "torch.float32",
    ctypes.c_double: "torch.float64",
    ctypes.c_uint32: "int",
    ctypes.c_int64: "torch.int64",
    ctypes.c_uint16: "torch.float16",
}

CTYPE_TO_CUTE = {
    ctypes.c_int: "cute.Int32",
    ctypes.c_float: "cute.Float32",
    ctypes.c_double: "cute.Float64",
    ctypes.c_uint32: "cute.Uint32",
    ctypes.c_int64: "cute.Int64",
    ctypes.c_uint16: "cute.Float16",
    ctypes.c_ulong: "cute.Uint64",
}

def ctype_to_cuda(ctype, name) -> str:
    if isinstance(ctype, type) and issubclass(ctype, ctypes._Pointer):
        base_type = getattr(ctype, "_type_", None)
        if base_type is None or base_type not in CTYPE_TO_CUDA:
            raise ValueError(
                f"Unsupported pointer base type: {base_type}. "
                "Please extend CTYPE_TO_CUDA mapping."
            )
        return f"{'const ' if name in CONST_HINTS else ''}{CTYPE_TO_CUDA[base_type]}*"

    if ctype not in CTYPE_TO_CUDA:
        raise ValueError(
            f"Unsupported scalar type: {ctype}. "
            "Please extend CTYPE_TO_CUDA mapping."
        )
    return CTYPE_TO_CUDA[ctype]

def ctype_to_mojo(ctype) -> str:
    if isinstance(ctype, type) and issubclass(ctype, ctypes._Pointer):
        base_type = getattr(ctype, "_type_", None)
        if base_type is None or base_type not in CTYPE_TO_MOJO:
            raise ValueError(
                f"Unsupported pointer base type: {base_type}. "
                "Please extend CTYPE_TO_MOJO mapping."
            )
        return f"UnsafePointer[{CTYPE_TO_MOJO[base_type]}]"

    if ctype not in CTYPE_TO_MOJO:
        raise ValueError(
            f"Unsupported scalar type: {ctype}. "
            "Please extend CTYPE_TO_MOJO mapping."
        )
    return CTYPE_TO_MOJO[ctype]

def ctype_to_torch(ctype, name) -> str:
    if isinstance(ctype, type) and issubclass(ctype, ctypes._Pointer):
        return f"{name}: torch.Tensor"

    if ctype in (ctypes.c_int, ctypes.c_uint32, ctypes.c_int64):
        return f"{name}: int"
    if ctype in (ctypes.c_float, ctypes.c_double):
        return f"{name}: float"

    raise ValueError(
        f"Unsupported type {ctype} for PyTorch mapping. "
        "Please extend CTYPE_TO_TORCH mapping."
    )

def ctype_to_cute(ctype, name) -> str:
    if isinstance(ctype, type) and issubclass(ctype, ctypes._Pointer):
        return f"{name}: cute.Tensor"

    if ctype not in CTYPE_TO_CUTE:
        raise ValueError(
            f"Unsupported scalar type: {ctype}. "
            "Please extend CTYPE_TO_CUTE mapping."
        )
    return f"{name}: {CTYPE_TO_CUTE[ctype]}"

def load_module(name: str, path: str):
    spec = util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")

    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module

def load_challenge(challenge_dir: str):
    base_url = "https://api.leetgpu.com/api/v1/core-files/challenge_base.py"
    base_dst = os.path.join(tempfile.gettempdir(), "challenge_base.py")
    urllib.request.urlretrieve(base_url, base_dst)

    sys.modules.setdefault("core", types.ModuleType("core")).__path__ = []

    load_module("core.challenge_base", base_dst)
    challenge = load_module("challenge", os.path.join(challenge_dir, "challenge.py"))

    return challenge.Challenge()

def generate_starter_cuda(sig, starter_file):
    arg_str = ", ".join(ctype_to_cuda(typ, name) + f" {name}" for name, typ in sig.items())
    include_half = "#include <cuda_fp16.h>\n" if "__half" in arg_str else ""
    
    device_pointers = [name for name, typ in sig.items() if isinstance(typ, type) and issubclass(typ, ctypes._Pointer)]
    comment = f"// {', '.join(device_pointers)} are device pointers" if device_pointers else ""
    
    code = f"""#include <cuda_runtime.h>
{include_half}
{comment}
extern "C" void solve({arg_str}) {{

}}"""
    with open(starter_file, "w") as f:
        f.write(code)

def generate_starter_mojo(sig, starter_file):
    arg_str = ", ".join(f"{name}: {ctype_to_mojo(typ)}" for name, typ in sig.items())
    
    device_pointers = [name for name, typ in sig.items() if isinstance(typ, type) and issubclass(typ, ctypes._Pointer)]
    comment = f"# {', '.join(device_pointers)} are device pointers" if device_pointers else ""
    
    code = f"""from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

{comment}
@export
def solve({arg_str}):
    pass"""

    with open(starter_file, "w") as f:
        f.write(code)

def generate_starter_pytorch(sig, starter_file):
    arg_str = ", ".join(ctype_to_torch(typ, name) for name, typ in sig.items())
    
    tensors = [name for name, typ in sig.items() if isinstance(typ, type) and issubclass(typ, ctypes._Pointer)]
    comment = f"# {', '.join(tensors)} are tensors on the GPU" if tensors else ""
    
    code = f"""import torch

{comment}
def solve({arg_str}):
    pass
"""
    with open(starter_file, "w") as f:
        f.write(code)

def generate_starter_triton(sig, starter_file):
    def ctype_to_triton(ctype, name):
        if isinstance(ctype, type) and issubclass(ctype, ctypes._Pointer):
            return f"{name}: torch.Tensor"
        if ctype in (ctypes.c_int, ctypes.c_uint32, ctypes.c_int64):
            return f"{name}: int"
        if ctype in (ctypes.c_float, ctypes.c_double):
            return f"{name}: float"
        raise ValueError(f"Unsupported type {ctype} for Triton mapping. Please extend ctype_to_triton mapping.")

    arg_str = ", ".join(ctype_to_triton(typ, name) for name, typ in sig.items())
    
    tensors = [name for name, typ in sig.items() if isinstance(typ, type) and issubclass(typ, ctypes._Pointer)]
    comment = f"# {', '.join(tensors)} are tensors on the GPU" if tensors else ""
    
    code = f"""import torch
import triton
import triton.language as tl

{comment}
def solve({arg_str}):
    pass
"""
    with open(starter_file, "w") as f:
        f.write(code)

def generate_starter_cute(sig, starter_file):
    arg_str = ", ".join(ctype_to_cute(typ, name) for name, typ in sig.items())
    
    tensors = [name for name, typ in sig.items() if isinstance(typ, type) and issubclass(typ, ctypes._Pointer)]
    comment = f"# {', '.join(tensors)} are tensors on the GPU" if tensors else ""
    
    code = f"""import cutlass
import cutlass.cute as cute

{comment}
@cute.jit
def solve({arg_str}):
    pass
"""
    with open(starter_file, "w") as f:
        f.write(code)

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/generate_starter_code.py path/to/challenge_dir")
        sys.exit(1)

    challenge_dir = sys.argv[1]

    if "easy" in (part.lower() for part in os.path.normpath(challenge_dir).split(os.sep)):
        print("Starter code generation script should not be used for 'easy' challenges.")
        sys.exit(1)

    starter_dir = os.path.join(challenge_dir, "starter")

    try:
        os.makedirs(starter_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating starter directory: {e}")
        sys.exit(1)

    challenge = load_challenge(challenge_dir)
    sig = challenge.get_solve_signature()

    generate_starter_cuda(sig, os.path.join(starter_dir, "starter.cu"))
    generate_starter_mojo(sig, os.path.join(starter_dir, "starter.mojo"))
    generate_starter_pytorch(sig, os.path.join(starter_dir, "starter.pytorch.py"))
    generate_starter_triton(sig, os.path.join(starter_dir, "starter.triton.py"))
    generate_starter_cute(sig, os.path.join(starter_dir, "starter.cute.py"))

if __name__ == "__main__":
    main()
