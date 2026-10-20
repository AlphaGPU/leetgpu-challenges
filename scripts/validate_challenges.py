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
TEST_GENERATORS = (
    "generate_example_test",
    "generate_functional_test",
    "generate_performance_test",
)
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


@dataclass(frozen=True)
class TestCase:
    lineno: int
    keys: frozenset[str]


FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def iter_scope_nodes(func: FunctionNode) -> Iterable[ast.AST]:
    """Yield nodes in a function body without descending into nested functions."""
    stack = list(reversed(func.body))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            stack.extend(reversed(list(ast.iter_child_nodes(node))))


class TestCaseResolver:
    """Statically resolve the dicts returned by a challenge's test generators.

    Follows return values through dict literals, dict(...) calls, lists, comprehensions,
    local variables (assignments, append/extend/+=, item assignment), nested helper
    functions and self._helper(...) methods. Resolution returns None when a test case
    is built in a way this resolver does not understand.
    """

    def __init__(self, class_node: ast.ClassDef):
        self.methods = {
            node.name: node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.active: set[tuple[int, str]] = set()

    def function_cases(self, scopes: list[FunctionNode]) -> list[TestCase] | None:
        func = scopes[-1]
        key = (id(func), "<return>")
        if key in self.active:
            return []
        self.active.add(key)
        try:
            cases: list[TestCase] = []
            for node in iter_scope_nodes(func):
                if isinstance(node, ast.Return) and node.value is not None:
                    resolved = self.expr_cases(node.value, scopes)
                    if resolved is None:
                        return None
                    cases.extend(resolved)
            return cases
        finally:
            self.active.discard(key)

    def expr_cases(self, expr: ast.expr, scopes: list[FunctionNode]) -> list[TestCase] | None:
        if isinstance(expr, ast.Dict):
            return self.dict_cases(expr, scopes)
        if isinstance(expr, (ast.List, ast.Tuple)):
            return self.concat_cases(expr.elts, scopes)
        if isinstance(expr, ast.Starred):
            return self.expr_cases(expr.value, scopes)
        if isinstance(expr, (ast.ListComp, ast.GeneratorExp)):
            return self.expr_cases(expr.elt, scopes)
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            return self.concat_cases([expr.left, expr.right], scopes)
        if isinstance(expr, ast.IfExp):
            return self.concat_cases([expr.body, expr.orelse], scopes)
        if isinstance(expr, ast.Name):
            return self.name_cases(expr.id, scopes)
        if isinstance(expr, ast.Call):
            return self.call_cases(expr, scopes)
        return None

    def concat_cases(
        self, exprs: Sequence[ast.expr], scopes: list[FunctionNode]
    ) -> list[TestCase] | None:
        cases: list[TestCase] = []
        for expr in exprs:
            resolved = self.expr_cases(expr, scopes)
            if resolved is None:
                return None
            cases.extend(resolved)
        return cases

    def dict_cases(self, node: ast.Dict, scopes: list[FunctionNode]) -> list[TestCase] | None:
        keys = set()
        unpacked = []
        for key, value in zip(node.keys, node.values):
            if key is None:
                unpacked.append(value)
            elif isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
            else:
                return None
        return self.merge_unpacked(node.lineno, keys, unpacked, scopes)

    def merge_unpacked(
        self, lineno: int, keys: set[str], unpacked: list[ast.expr], scopes: list[FunctionNode]
    ) -> list[TestCase] | None:
        cases = [TestCase(lineno, frozenset(keys))]
        for value in unpacked:
            resolved = self.expr_cases(value, scopes)
            if not resolved:
                return None
            cases = [
                TestCase(case.lineno, case.keys | other.keys)
                for case in cases
                for other in resolved
            ]
        return cases

    def dict_call_cases(self, node: ast.Call, scopes: list[FunctionNode]) -> list[TestCase] | None:
        keys = {keyword.arg for keyword in node.keywords if keyword.arg is not None}
        unpacked = list(node.args)
        unpacked.extend(keyword.value for keyword in node.keywords if keyword.arg is None)
        return self.merge_unpacked(node.lineno, keys, unpacked, scopes)

    def call_cases(self, node: ast.Call, scopes: list[FunctionNode]) -> list[TestCase] | None:
        func = node.func
        if isinstance(func, ast.Name) and func.id == "dict":
            return self.dict_call_cases(node, scopes)
        if isinstance(func, ast.Name):
            for depth in range(len(scopes) - 1, -1, -1):
                for child in iter_scope_nodes(scopes[depth]):
                    if (
                        isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and child.name == func.id
                    ):
                        return self.function_cases(scopes[: depth + 1] + [child])
            return None
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
            and func.attr in self.methods
        ):
            return self.function_cases([self.methods[func.attr]])
        return None

    def name_cases(self, name: str, scopes: list[FunctionNode]) -> list[TestCase] | None:
        for depth in range(len(scopes) - 1, -1, -1):
            resolved = self.local_name_cases(name, scopes[: depth + 1])
            if resolved is not None:
                return resolved
        return None

    def local_name_cases(self, name: str, scopes: list[FunctionNode]) -> list[TestCase] | None:
        key = (id(scopes[-1]), name)
        if key in self.active:
            return []
        self.active.add(key)
        try:
            sources: list[ast.expr] = []
            extra_keys: set[str] = set()
            for node in iter_scope_nodes(scopes[-1]):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == name:
                            sources.append(node.value)
                        elif is_item_of(target, name):
                            item_key = target.slice
                            if not (
                                isinstance(item_key, ast.Constant)
                                and isinstance(item_key.value, str)
                            ):
                                return None
                            extra_keys.add(item_key.value)
                elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                    target = node.target
                    if isinstance(target, ast.Name) and target.id == name and node.value:
                        sources.append(node.value)
                elif isinstance(node, (ast.For, ast.comprehension)):
                    if isinstance(node.target, ast.Name) and node.target.id == name:
                        sources.append(node.iter)
                elif isinstance(node, ast.Call) and is_method_of(node.func, name):
                    method = node.func.attr
                    if method in ("append", "extend") and node.args:
                        sources.append(node.args[0])
                    elif method == "insert" and len(node.args) == 2:
                        sources.append(node.args[1])
                    elif method == "update":
                        updates = self.dict_call_cases(node, scopes)
                        if updates is None:
                            return None
                        for case in updates:
                            extra_keys |= case.keys
            if not sources:
                return None
            cases = self.concat_cases(sources, scopes)
            if cases is None:
                return None
            return [TestCase(case.lineno, case.keys | extra_keys) for case in cases]
        finally:
            self.active.discard(key)


def is_item_of(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == name
    )


def is_method_of(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == name
    )


def reference_impl_params(func: FunctionNode) -> tuple[set[str], set[str], bool]:
    """Return (required, accepted, accepts **kwargs) parameter names, excluding self."""
    args = func.args
    positional = args.posonlyargs + args.args
    if positional and positional[0].arg == "self":
        positional = positional[1:]
    first_default = len(positional) - len(args.defaults)
    required = {arg.arg for arg in positional[:first_default]}
    required |= {
        arg.arg for arg, default in zip(args.kwonlyargs, args.kw_defaults) if default is None
    }
    accepted = {arg.arg for arg in positional + args.kwonlyargs}
    return required, accepted, args.kwarg is not None


def validate_test_case_keys(challenge_py: Path, challenge_class: ast.ClassDef) -> list[Issue]:
    resolver = TestCaseResolver(challenge_class)
    reference_impl = resolver.methods.get("reference_impl")
    if reference_impl is None:
        return []
    required, accepted, accepts_kwargs = reference_impl_params(reference_impl)

    issues = []
    for generator in TEST_GENERATORS:
        method = resolver.methods.get(generator)
        if method is None:
            continue
        cases = resolver.function_cases([method])
        if cases is None:
            issues.append(
                Issue(
                    "ERROR",
                    challenge_py,
                    f"{generator}: could not statically resolve the keys of its test dicts; "
                    "build them from dict literals or dict(...) calls",
                )
            )
            continue
        if not cases:
            issues.append(Issue("ERROR", challenge_py, f"{generator}: returns no test dicts"))
            continue
        for case in sorted(set(cases), key=lambda case: (case.lineno, sorted(case.keys))):
            missing = sorted(required - case.keys)
            unexpected = [] if accepts_kwargs else sorted(case.keys - accepted)
            problems = []
            if missing:
                problems.append("missing reference_impl parameter(s): " + ", ".join(missing))
            if unexpected:
                problems.append("unexpected key(s): " + ", ".join(unexpected))
            if problems:
                message = f"{generator}: test dict at line {case.lineno} " + "; ".join(problems)
                issues.append(Issue("ERROR", challenge_py, message))
    return issues


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

            issues.extend(validate_test_case_keys(challenge_py, challenge_class))

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
