import os
import glob
import json
import re

def generate_notebook(challenge_dir, output_dir):
    parts = challenge_dir.strip('/').split('/')
    level = parts[-2]
    name = parts[-1]
    
    # Check if files exist
    challenge_py_path = os.path.join(challenge_dir, "challenge.py")
    challenge_html_path = os.path.join(challenge_dir, "challenge.html")
    starter_dir = os.path.join(challenge_dir, "starter")
    
    if not os.path.exists(challenge_py_path) or not os.path.exists(challenge_html_path):
        return
        
    print(f"Migrating {level} {name}...")
    
    cells = []
    
    # 1. Config cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "config_cell"},
        "outputs": [],
        "source": [
            "# Change this to your preferred framework (e.g., 'cuda', 'pytorch', 'triton', 'jax', 'mojo')\n",
            "EVAL_LANG = 'cuda'\n",
            "\n",
            "SAVE_GPU = True\n"
        ]
    })
    
    # 2. Markdown description
    with open(challenge_html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
        
    # Replace LaTeX delimiters with standard markdown/mathjax delimiters
    html_content = html_content.replace("\\(", "$").replace("\\)", "$")
    html_content = html_content.replace("\\[", "$$").replace("\\]", "$$")
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "desc_cell"},
        "source": [html_content]
    })
    
    # 3. Starter templates (hidden cells)
    if os.path.exists(starter_dir):
        for starter_file in sorted(os.listdir(starter_dir)):
            if starter_file.startswith("starter."):
                ext = starter_file[len("starter."):]
                
                # Header mapping
                header_map = {
                    "cu": "# CUDA",
                    "cute.py": "# CUTE",
                    "jax.py": "# JAX",
                    "mojo": "# MOJO",
                    "pytorch.py": "# Torch",
                    "triton.py": "# Triton"
                }
                header_text = header_map.get(ext, f"# {ext.upper()}")
                
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [header_text]
                })
                
                # Default output filenames
                out_filename = "solution.cu" if ext == "cu" else f"solution.{ext}"
                if out_filename.endswith(".py"):
                    out_filename = "solution.py"
                    
                with open(os.path.join(starter_dir, starter_file), 'r', encoding='utf-8') as f:
                    starter_content = f.read()
                    
                cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {
                        "id": f"starter_{ext.replace('.', '_')}",
                        "cellView": "form",
                        "collapsed": True
                    },
                    "outputs": [],
                    "source": [
                        f"%%writefile {out_filename}\n",
                        starter_content
                    ]
                })
    
    # 4. Challenge Base & Challenge logic
    base_py_path = os.path.join("challenges", "core", "challenge_base.py")
    with open(base_py_path, 'r', encoding='utf-8') as f:
        base_content = f.read()
        
    with open(challenge_py_path, 'r', encoding='utf-8') as f:
        challenge_content = f.read()
        
    # Remove the import statement
    challenge_content = re.sub(r"from core\.challenge_base import ChallengeBase\n?", "", challenge_content)
    
    combined_challenge = (
        "# --- Core Challenge Base ---\n" +
        base_content + "\n\n" +
        "# --- Challenge Logic ---\n" +
        challenge_content + "\n\n" +
        "ch = Challenge()\n"
    )
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Evaluate Setup"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "challenge_logic"},
        "outputs": [],
        "source": [line + "\n" for line in combined_challenge.split("\n")]
    })
    
    # 5. Evaluator script
    eval_script = """import os
import time
import ctypes
import torch

class Evaluate:
    @staticmethod
    def eval_cuda(ch):
        # 1. Compile a fresh uniquely named library
        so_filename = f'solution_func_{int(time.time())}.so'
        os.system(f'nvcc -shared -Xcompiler -fPIC -O3 solution.cu -o {so_filename}')
        lib = ctypes.CDLL(f'./{so_filename}')
        
        # 2. Extract signature and set argtypes
        signature = ch.get_solve_signature()
        lib.solve.argtypes = [arg_info[0] for arg_info in signature.values()]
        
        Evaluate._run_tests(ch, signature, lambda kwargs: lib.solve(*Evaluate._build_cuda_args(kwargs, signature)))

    @staticmethod
    def eval_python(ch):
        import importlib.util
        import sys
        
        spec = importlib.util.spec_from_file_location("solution", "solution.py")
        solution = importlib.util.module_from_spec(spec)
        sys.modules["solution"] = solution
        spec.loader.exec_module(solution)
        
        signature = ch.get_solve_signature()
        Evaluate._run_tests(ch, signature, lambda kwargs: Evaluate._run_python(solution, kwargs))

    @staticmethod
    def _run_python(solution, kwargs):
        solution.solve(**kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def eval_mojo(ch):
        print("Mojo evaluation is currently executed via a separate runner or wrapper.")
        print("Ensure you have the mojo compiler installed and use 'mojo build solution.mojo' + ctypes/ffi,")
        print("or run an external python bridge. This is a stub.")

    @staticmethod
    def _build_cuda_args(kwargs, signature):
        cuda_args = []
        for k, (arg_type, dir_type) in signature.items():
            val = kwargs[k]
            if isinstance(val, torch.Tensor):
                cuda_args.append(ctypes.cast(val.data_ptr(), arg_type))
            else:
                cuda_args.append(arg_type(val))
        return cuda_args

    @staticmethod
    def _run_tests(ch, signature, run_fn):
        print("=== Running Functional Tests ===")
        functional_tests = ch.generate_functional_test()
        all_passed = True
        
        for i, test in enumerate(functional_tests):
            ref_kwargs = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in test.items()}
            test_kwargs = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in test.items()}
            
            # Run Reference
            ch.reference_impl(**ref_kwargs)
            
            # Run implementation
            run_fn(test_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Verify outputs
            match = True
            for k, (_, dir_type) in signature.items():
                if dir_type == "out":
                    if not torch.allclose(ref_kwargs[k], test_kwargs[k], atol=ch.atol, rtol=ch.rtol):
                        match = False
                        print(f"❌ Test {i+1}/{len(functional_tests)} Failed on output '{k}'")
                        break
            
            if match:
                print(f"✅ Test {i+1}/{len(functional_tests)} Passed")
            else:
                all_passed = False
                break
                
        if all_passed:
            print("\\n🎉 All functional tests passed!")
            return True
        else:
            return False
"""
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "evaluator", "cellView": "form", "collapsed": True},
        "outputs": [],
        "source": [line + "\n" for line in eval_script.split("\n")]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Evaluation code"]
    })
    
    # 6. Run and Disconnect runtime cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "disconnect"},
        "outputs": [],
        "source": [
            "# Run the evaluator based on configuration\n",
            "if EVAL_LANG == 'cuda':\n",
            "    Evaluate.eval_cuda(ch)\n",
            "elif EVAL_LANG in ['pytorch', 'triton', 'jax', 'cute']:\n",
            "    Evaluate.eval_python(ch)\n",
            "elif EVAL_LANG == 'mojo':\n",
            "    Evaluate.eval_mojo(ch)\n",
            "else:\n",
            "    print(f\"Unknown language {EVAL_LANG}\")\n",
            "\n",
            "# Disconnect runtime to save Colab resources\n",
            "if SAVE_GPU:\n",
            "    from google.colab import runtime\n",
            "    runtime.unassign()\n"
        ]
    })
    
    notebook = {
      "nbformat": 4,
      "nbformat_minor": 0,
      "metadata": {
        "accelerator": "GPU",
        "colab": {
          "gpuType": "T4",
          "provenance": []
        }
      },
      "cells": cells
    }
    
    out_dir_level = os.path.join(output_dir, level)
    out_file = os.path.join(out_dir_level, f"{name}.ipynb")
    os.makedirs(out_dir_level, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    out_dir = "challenges/colab_exports"
    
    challenges_glob = glob.glob("challenges/*/*")
    for challenge_dir in challenges_glob:
        if os.path.isdir(challenge_dir) and challenge_dir != "challenges/colab_exports":
            generate_notebook(challenge_dir, out_dir)
