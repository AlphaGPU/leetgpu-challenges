#!/usr/bin/env python3
"""
Development helper to upsert a challenge and submit a solution via /ws/submit.

Usage:
    python scripts/run_challenge.py /path/to/challenges/easy/1_vector_add

Env vars:
    SERVICE_URL       - API base URL with protocol (default: http://localhost:8080)
    LEETGPU_API_KEY   - required, Bearer token
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import websocket
from update_challenges import LEETGPU_API_KEY, SERVICE_URL, load_challenge, update_challenge

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def find_solution_file(challenge_dir: Path, language: str) -> tuple[str, str]:
    language_to_extension = {
        "cuda": "cu",
        "mojo": "mojo",
        "pytorch": "pytorch.py",
        "cute": "cute.py",
        "triton": "triton.py",
        "jax": "jax.py",
    }
    solution_file = challenge_dir / "solution" / f"solution.{language_to_extension[language]}"
    if not solution_file.exists():
        raise FileNotFoundError(
            f"No solution file found for {language}. Add a solution/solution.{language}.py file."
        )
    return solution_file.name, solution_file.read_text()


def submit_solution(
    ws_url: str,
    api_key: str,
    challenge_id: int,
    file_name: str,
    content: str,
    language: str,
    gpu: str,
    action: str,
    public: bool,
) -> bool:

    headers = [f"Authorization: Bearer {api_key}"] if api_key else []
    ws = websocket.create_connection(ws_url, header=headers, timeout=120)
    try:
        submission = {
            "action": action,
            "submission": {
                "files": [{"name": file_name, "content": content}],
                "language": language,
                "gpu": gpu,
                "mode": "accelerated",
                "public": public,
                "challenge_id": challenge_id,
            },
        }
        ws.send(json.dumps(submission))
        logger.info("Submitted %s to challenge %s", file_name, challenge_id)

        while True:
            msg = ws.recv()
            if not msg:
                continue
            data = json.loads(msg)
            status = data.get("status")
            output = data.get("output")
            logger.info("Status: %s | Output: %s", status, output)
            if status in {"success", "error", "timeout", "oom", "interrupted"}:
                return status == "success"
    finally:
        ws.close()


def main() -> int:
    if not LEETGPU_API_KEY:
        logger.error("LEETGPU_API_KEY environment variable is required")
        return 1

    parser = argparse.ArgumentParser(
        description="Upsert a challenge and submit a solution via API."
    )
    parser.add_argument("challenge_path", type=Path, help="Path to the challenge directory")
    parser.add_argument("--language", default="cuda", help="Language (default: cuda)")
    parser.add_argument(
        "--gpu", default="NVIDIA TESLA T4", help="GPU name (default: NVIDIA TESLA T4)"
    )
    parser.add_argument(
        "--action", default="run", choices=["run", "submit"], help="Action (run or submit)"
    )
    args = parser.parse_args()

    # Upsert challenge
    try:
        payload = load_challenge(args.challenge_path)
    except Exception as e:
        logger.error("Failed to load challenge: %s", e)
        return 1

    if not update_challenge(SERVICE_URL, payload, LEETGPU_API_KEY):
        logger.error("Upsert failed")
        return 1

    # Submit solution
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
        challenge_id=payload["id"],
        file_name=file_name,
        content=content,
        language=args.language,
        gpu=args.gpu,
        action=args.action,
        public=False,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
