#!/usr/bin/env python3
"""
Deploy challenges to LeetGPU.com

Environment variables:
    SERVICE_URL - API service URL (default: localhost:8080)
    LEETGPU_API_KEY - API key for authentication (required)
"""

import argparse
import importlib.util
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SERVICE_URL = os.getenv("SERVICE_URL", "localhost:8080")
LEETGPU_API_KEY = os.getenv("LEETGPU_API_KEY")

GPUS = ["NVIDIA H100", "NVIDIA H200", "NVIDIA TESLA T4", "NVIDIA B200", "NVIDIA A100-80GB"]

def extract_id(name: str) -> int:
    match = re.match(r"^(\d+)_", name)
    if not match:
        raise ValueError(f"Directory name must start with number: {name}")
    return int(match.group(1))

def get_difficulty(path: Path) -> str:
    s = str(path).lower()
    for d in ("easy", "medium", "hard"):
        if d in s:
            return d
    return "easy"

def get_language(filename: str) -> Optional[str]:
    if filename == "starter.cu":
        return "cuda"
    if filename == "starter.mojo":
        return "mojo"
    if filename.startswith("starter.") and filename.endswith(".py"):
        parts = filename.split(".")
        if len(parts) == 3:
            return parts[1]
    return None

def load_challenge(problem_dir: Path) -> Dict:
    logger.info("Loading %s", problem_dir)
    
    problem_id = extract_id(problem_dir.name)
    
    spec_path = problem_dir / "challenge.html"
    if not spec_path.exists():
        spec_path = problem_dir / "problem.html"
    if not spec_path.exists():
        raise FileNotFoundError(f"No spec file in {problem_dir}")
    
    challenge_path = problem_dir / "challenge.py"
    if not challenge_path.exists():
        raise FileNotFoundError(f"No challenge.py in {problem_dir}")
    
    challenges_dir = problem_dir.parent.parent
    sys.path.insert(0, str(challenges_dir))
    
    try:
        spec = importlib.util.spec_from_file_location("challenge", challenge_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        challenge = module.Challenge()
        title = challenge.name
        access_tier = challenge.access_tier
    finally:
        sys.path.remove(str(challenges_dir))
        if "challenge" in sys.modules:
            del sys.modules["challenge"]
    
    starter_code = []
    starter_dir = problem_dir / "starter"
    if starter_dir.exists():
        for f in starter_dir.iterdir():
            if f.is_file() and (lang := get_language(f.name)):
                starter_code.append({
                    "language": lang,
                    "fileName": f.name,
                    "fileContent": f.read_text()
                })
    
    return {
        "id": problem_id,
        "title": title,
        "spec": spec_path.read_text(),
        "challengeCode": challenge_path.read_text(),
        "difficultyLevel": get_difficulty(problem_dir),
        "accessTier": access_tier,
        "gpus": GPUS, # TODO: get from challenge.py or API
        "starterCode": starter_code,
    }

def update_challenge(service_url: str, payload: Dict, api_key: str) -> bool:
    url = f"http://{service_url.rstrip('/')}/api/v1/challenges"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        logger.info("Updated challenge %s: %s", payload["id"], payload["title"])
        return True
    except Exception as e:
        logger.error("Failed challenge %s: %s", payload["id"], e)
        return False

def main():
    if not LEETGPU_API_KEY:
        logger.error("LEETGPU_API_KEY environment variable is required")
        return 1
    
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?")
    args = parser.parse_args()
    
    if args.path:
        dirs = [args.path]
    else:
        base = Path(__file__).parent.parent / "challenges"
        dirs = [d for diff in base.iterdir() if diff.is_dir()
                for d in diff.iterdir() if d.is_dir()]
    
    success = fail = 0
    for d in sorted(dirs):
        try:
            payload = load_challenge(d)
            if update_challenge(SERVICE_URL, payload, LEETGPU_API_KEY):
                success += 1
            else:
                fail += 1
        except Exception as e:
            logger.error("Failed %s: %s", d, e)
            fail += 1
    
    logger.info("Summary: %d succeeded, %d failed", success, fail)
    return 0 if fail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
