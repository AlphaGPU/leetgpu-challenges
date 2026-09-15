#!/usr/bin/env python3
"""
Validate challenge structure and metadata without importing challenge dependencies.

This script is intentionally AST-based so it can run in CI without PyTorch, CUDA,
JAX, or other challenge runtime dependencies installed.
"""

import argparse
import ast
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DIFFICULTIES = ("easy", "medium", "hard")
REQUIRED_FILES = ("challenge.py", "challenge.html")
REQUIRED_METADATA = ("name", "atol", "rtol", "num_gpus", "access_tier")
REQUIRED_METHODS = (
    "reference_impl",
    "get_solve_signature",
    "generate_example_test",
    "generate_functional_test",
    "generate_performance_test",
)
SUPPORTED_STARTERS = {
    "starter.cu",
    "starter.cute.py",
    "starter.jax.py",
    "starter.mojo",
    "starter.pytorch.py",
    "starter.triton.py",
}
CHALLENGE_ID_RE = re.compile(r"^(?P<id>\d+)_")
CHALLENGE_DIR_RE = re.compile(r"^(?P<id>\d+)_[a-z0-9_]+$")


@dataclass(frozen=True)
class Issue:
    severity: str
    path: Path
    message: str

    def format(self, root: Path) -> str:
        try:
            display_path = self.path.relative_to(root)
        except ValueError:
            display_path = self.path
        return f"{self.severity}: {display_path}: {self.message}"


def iter_challenge_dirs(challenges_root: Path) -> Iterable[Path]:
    for difficulty in DIFFICULTIES:
        difficulty_dir = challenges_root / difficulty
        if not difficulty_dir.is_dir():
            continue
        yield from sorted(path for path in difficulty_dir.iterdir() if path.is_dir())


def get_challenge_id(challenge_dir: Path) -> int | None:
    match = CHALLENGE_ID_RE.match(challenge_dir.name)
    if not match:
        return None
    return int(match.group("id"))


def find_challenge_class(challenge_py: Path) -> ast.ClassDef | None:
    try:
        tree = ast.parse(challenge_py.read_text(encoding="utf-8"), filename=str(challenge_py))
    except SyntaxError:
        return None

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Challenge":
            return node
    return None


def class_assignments(class_node: ast.ClassDef) -> set[str]:
    assignments = set()
    for node in class_node.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assignments.add(node.target.id)
    return assignments


def class_methods(class_node: ast.ClassDef) -> set[str]:
    return {
        node.name
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def validate_challenge_dir(challenge_dir: Path, strict_starters: bool) -> list[Issue]:
    issues = []

    if not CHALLENGE_DIR_RE.match(challenge_dir.name):
        severity = "ERROR" if get_challenge_id(challenge_dir) is None else "WARN"
        issues.append(
            Issue(
                severity,
                challenge_dir,
                "directory name should match '<number>_<lowercase_slug>'",
            )
        )

    for filename in REQUIRED_FILES:
        if not (challenge_dir / filename).is_file():
            issues.append(Issue("ERROR", challenge_dir, f"missing required file: {filename}"))

    challenge_py = challenge_dir / "challenge.py"
    if challenge_py.is_file():
        challenge_class = find_challenge_class(challenge_py)
        if challenge_class is None:
            issues.append(Issue("ERROR", challenge_py, "missing parseable Challenge class"))
        else:
            assignments = class_assignments(challenge_class)
            methods = class_methods(challenge_class)

            for name in REQUIRED_METADATA:
                if name not in assignments:
                    issues.append(Issue("ERROR", challenge_py, f"missing metadata: {name}"))

            for name in REQUIRED_METHODS:
                if name not in methods:
                    issues.append(Issue("ERROR", challenge_py, f"missing method: {name}"))

    starter_dir = challenge_dir / "starter"
    if not starter_dir.is_dir():
        issues.append(Issue("ERROR", challenge_dir, "missing starter directory"))
        return issues

    starter_files = {path.name for path in starter_dir.iterdir() if path.is_file()}
    unknown_starters = sorted(starter_files - SUPPORTED_STARTERS)
    missing_starters = sorted(SUPPORTED_STARTERS - starter_files)

    for filename in unknown_starters:
        issues.append(Issue("ERROR", starter_dir / filename, "unsupported starter filename"))

    if missing_starters:
        severity = "ERROR" if strict_starters else "WARN"
        issues.append(
            Issue(
                severity,
                starter_dir,
                "missing starter files: " + ", ".join(missing_starters),
            )
        )

    return issues


def validate_unique_ids(challenge_dirs: Sequence[Path]) -> list[Issue]:
    by_id: dict[int, list[Path]] = defaultdict(list)
    for challenge_dir in challenge_dirs:
        challenge_id = get_challenge_id(challenge_dir)
        if challenge_id is not None:
            by_id[challenge_id].append(challenge_dir)

    issues = []
    for challenge_id, paths in sorted(by_id.items()):
        if len(paths) > 1:
            joined_paths = ", ".join(str(path) for path in paths)
            issues.append(
                Issue("ERROR", paths[0], f"duplicate challenge id {challenge_id}: {joined_paths}")
            )
    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LeetGPU challenge structure.")
    parser.add_argument(
        "--challenges-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "challenges",
        help="Path to the challenges directory.",
    )
    parser.add_argument(
        "--strict-starters",
        action="store_true",
        help="Treat missing starter templates as errors instead of warnings.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    challenges_root = args.challenges_root.resolve()
    repo_root = challenges_root.parent

    if not challenges_root.is_dir():
        print(f"ERROR: {challenges_root}: challenges root does not exist", file=sys.stderr)
        return 1

    challenge_dirs = list(iter_challenge_dirs(challenges_root))
    issues = validate_unique_ids(challenge_dirs)
    for challenge_dir in challenge_dirs:
        issues.extend(validate_challenge_dir(challenge_dir, args.strict_starters))

    for issue in issues:
        stream = sys.stderr if issue.severity == "ERROR" else sys.stdout
        print(issue.format(repo_root), file=stream)

    error_count = sum(issue.severity == "ERROR" for issue in issues)
    warning_count = sum(issue.severity == "WARN" for issue in issues)
    print(
        f"Validated {len(challenge_dirs)} challenges: "
        f"{error_count} errors, {warning_count} warnings"
    )

    return 1 if error_count else 0


if __name__ == "__main__":
    sys.exit(main())
