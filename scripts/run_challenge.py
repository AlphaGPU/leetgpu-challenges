#!/usr/bin/env python3
"""
Development helper to submit a solution via /ws/submit.

Usage:
    python scripts/run_challenge.py /path/to/challenges/easy/1_vector_add

Env vars:
    SERVICE_URL       - API base URL with protocol (default: http://localhost:8080)
    LEETGPU_API_KEY   - required, Bearer token
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import websocket


def load_challenge(challenge_dir: Path):
    """Import challenge_dir/challenge.py and return an instantiated Challenge.

    Mirrors the loading dance in scripts/update_challenges.py: the parent of
    the `easy|medium|hard` directory is added to sys.path so the module can
    `from core.challenge_base import ChallengeBase`.
    """
    challenge_py = challenge_dir / "challenge.py"
    challenges_root = challenge_dir.parent.parent
    sys.path.insert(0, str(challenges_root))
    try:
        spec = importlib.util.spec_from_file_location("challenge", challenge_py)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Challenge()
    finally:
        sys.path.remove(str(challenges_root))
        sys.modules.pop("challenge", None)


SERVICE_URL = os.getenv("SERVICE_URL", "http://localhost:8080")
LEETGPU_API_KEY = os.getenv("LEETGPU_API_KEY")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def find_solution_file(challenge_dir: Path, language: str) -> tuple[str, str]:
    language_to_extension = {
        "cuda": "cu",
        "mojo": "mojo",
        "pytorch": "py",
        "cute": "py",
        "triton": "py",
        "jax": "py",
    }
    ext = language_to_extension[language]
    # prefer a language-tagged filename (solution.triton.py, solution.jax.py, ...)
    # so multiple python-based languages can coexist in the same challenge dir.
    candidates = [
        challenge_dir / "solution" / f"solution.{language}.{ext}",
        challenge_dir / "solution" / f"solution.{ext}",
    ]
    for path in candidates:
        if path.exists():
            # the server expects the filename to match the language's canonical solution name.
            canonical_name = f"solution.{ext}"
            return canonical_name, path.read_text()
    raise FileNotFoundError(
        f"No solution file found for {language}. " f"Tried: {', '.join(str(p) for p in candidates)}"
    )


def submit_solution(
    ws_url: str,
    api_key: str,
    challenge_code: str,
    file_name: str,
    content: str,
    language: str,
    gpu: str,
    gpu_count: int,
    action: str,
    public: bool,
) -> bool:

    ws = websocket.create_connection(ws_url, timeout=120)
    try:
        submission = {
            "action": action,
            "token": api_key,
            "submission": {
                "files": [{"name": file_name, "content": content}],
                "language": language,
                "gpu": gpu,
                "gpuCount": gpu_count,
                "mode": "accelerated",
                "public": public,
                "challengeCode": challenge_code,
            },
        }
        ws.send(json.dumps(submission))
        logger.info("Submitted %s", file_name)

        while True:
            msg = ws.recv()
            if not msg:
                continue
            data = json.loads(msg)
            status = data.get("status")
            output = data.get("output") or ""
            typ = data.get("type") or ""
            if output:
                sys.stdout.write(f"[{status}/{typ}] {output}")
                if not output.endswith("\n"):
                    sys.stdout.write("\n")
                sys.stdout.flush()
            if status in {
                "success",
                "error",
                "timeout",
                "oom",
                "interrupted",
                "test-case-failed",
                "compilation-failed",
                "tampering-detected",
                "out-of-memory",
            }:
                return status == "success"
    finally:
        ws.close()


def main() -> int:
    if not LEETGPU_API_KEY:
        logger.error("LEETGPU_API_KEY environment variable is required")
        return 1

    parser = argparse.ArgumentParser(description="Submit a solution via WebSocket API.")
    parser.add_argument("challenge_path", type=Path, help="Path to the challenge directory")
    parser.add_argument("--language", default="cuda", help="Language (default: cuda)")
    parser.add_argument("--gpu", default="T4", help="GPU name (default: T4)")
    parser.add_argument(
        "--action", default="run", choices=["run", "submit"], help="Action (run or submit)"
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=None,
        help="Number of GPUs (default: auto-detected from challenge.py num_gpus)",
    )
    args = parser.parse_args()

    challenge_py = args.challenge_path / "challenge.py"
    if not challenge_py.exists():
        logger.error("No challenge.py found in %s", args.challenge_path)
        return 1
    challenge_code = challenge_py.read_text()
    try:
        challenge = load_challenge(args.challenge_path)
    except Exception as e:
        logger.error("Failed to load challenge module: %s", e)
        return 1
    gpu_count = args.gpu_count if args.gpu_count is not None else getattr(challenge, "num_gpus", 1)
    logger.info("Using gpu_count=%d", gpu_count)

    try:
        file_name, content = find_solution_file(args.challenge_path, args.language)
    except Exception as e:
        logger.error("Failed to find solution file: %s", e)
        return 1

    # Convert http(s) URL to ws(s) URL
    ws_url = SERVICE_URL.replace("https://", "wss://").replace("http://", "ws://")
    ok = submit_solution(
        ws_url=f"{ws_url.rstrip('/')}/api/v1/ws/submit",
        api_key=LEETGPU_API_KEY,
        challenge_code=challenge_code,
        file_name=file_name,
        content=content,
        language=args.language,
        gpu=args.gpu,
        gpu_count=gpu_count,
        action=args.action,
        public=False,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
