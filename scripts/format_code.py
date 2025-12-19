#!/usr/bin/env python3
"""
Format / lint Python files with ruff + black, only on files not ignored by git.

Date: create on 07/12/2025
Author: Yang Zhou, zyaztec@gmail.com

Usage:
  python format_python.py [--check] [--no-install] [path ...]

  --check       Run in check/diff mode (no files will be modified).
  --no-install  Do not auto-install ruff/black; expect them to already exist.
  path ...      Optional paths (files or directories). If omitted, the repo root is used.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Format all non-ignored Python files with ruff + black.",
        add_help=True,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run tools in check/diff mode without modifying files.",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Skip automatic pip installs; expect ruff/black to already exist.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Paths to include (files or directories). Defaults to repo root.",
    )
    return parser.parse_args()


def get_repo_root() -> Path:
    """Return the git repo root (using `git rev-parse`)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        print(
            "Error: This script must be run inside a git repository "
            f"(git rev-parse failed with: {exc.output.decode().strip()})",
            file=sys.stderr,
        )
        sys.exit(1)
    return Path(out.decode().strip()).resolve()


def ensure_tools(should_install: bool) -> None:
    """Ensure ruff and black are available (optionally installing them)."""
    if should_install:
        print("Installing/Updating ruff and black via pip...", file=sys.stderr)
        # Use the same interpreter running this script
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-U",
            "ruff>=0.6.0",
            "black>=24.0.0",
        ]
        subprocess.check_call(cmd)
        return

    # --no-install: check that both commands are available
    missing: List[str] = []
    for tool in ("ruff", "black"):
        if shutil.which(tool) is None:
            missing.append(tool)

    if missing:
        print(
            "Missing dependencies: "
            + ", ".join(missing)
            + "\nRe-run without --no-install to install them automatically.",
            file=sys.stderr,
        )
        sys.exit(1)


def is_under(path: Path, parent: Path) -> bool:
    """Return True if `path` is the same as or inside `parent`."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def filter_files_under(
    all_files: Iterable[Path],
    targets: List[Path],
) -> List[Path]:
    """Keep only files that are under any of the target paths."""
    if not targets:
        return list(all_files)

    filtered: List[Path] = []
    for f in all_files:
        if any(is_under(f, t) for t in targets):
            filtered.append(f)
    return filtered


def get_non_ignored_py_files(repo_root: Path, target_paths: List[Path]) -> List[Path]:
    """
    Use git to list all non-ignored files, then filter to .py
    and to those under target_paths.
    """
    # git ls-files:
    #   --cached   -> tracked
    #   --others   -> untracked but not ignored
    #   --exclude-standard -> respect .gitignore, .git/info/exclude, global ignore
    cmd = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
    out = subprocess.check_output(cmd, cwd=repo_root)
    lines = out.decode().splitlines()

    all_py_files: List[Path] = []
    for rel in lines:
        p = (repo_root / rel).resolve()
        if p.suffix == ".py" and p.exists():
            all_py_files.append(p)

    py_files = filter_files_under(all_py_files, target_paths)
    return py_files


def chunked(seq: List[Path], size: int = 200) -> Iterable[List[Path]]:
    """Yield chunks of the list to avoid extremely long command lines."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def run_command(cmd: List[str], allow_failure: bool = False) -> int:
    """Run a command and propagate non-zero exit codes.

    Args:
        cmd: The command to run.
        allow_failure: If True, don't raise an exception on non-zero exit codes.

    Returns:
        The exit code of the command.
    """
    # Print the command for debugging (optional, can be removed)
    print("+ " + " ".join(cmd))
    if allow_failure:
        return subprocess.call(cmd)
    else:
        subprocess.check_call(cmd)
        return 0


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()

    # Resolve target paths
    if args.paths:
        target_paths: List[Path] = []
        for p in args.paths:
            resolved = Path(p).resolve()
            if resolved.exists():
                target_paths.append(resolved)
            else:
                print(
                    f"Warning: path does not exist, skipping: {resolved}",
                    file=sys.stderr,
                )
    else:
        target_paths = [repo_root]

    if not target_paths:
        print("No valid paths provided; nothing to do.")
        sys.exit(0)

    ensure_tools(should_install=not args.no_install)

    py_files = get_non_ignored_py_files(repo_root, target_paths)

    if not py_files:
        print(
            "No non-ignored Python files found under:",
            ", ".join(map(str, target_paths)),
        )
        sys.exit(0)

    print(f"Found {len(py_files)} Python files to process.")

    exit_code = 0  # 用于 --check 模式的最终退出码

    for chunk in chunked(py_files):
        paths_str = [str(p) for p in chunk]

        if args.check:
            ruff_cmd = [sys.executable, "-m", "ruff", "check", "--diff", *paths_str]
            black_cmd = [sys.executable, "-m", "black", "--check", "--diff", *paths_str]

            ruff_ret = run_command(ruff_cmd, allow_failure=True)
            black_ret = run_command(black_cmd, allow_failure=True)

            if ruff_ret != 0 or black_ret != 0:
                exit_code = 1
        else:
            ruff_cmd = [sys.executable, "-m", "ruff", "check", "--fix", *paths_str]
            black_cmd = [sys.executable, "-m", "black", *paths_str]

            ruff_ret = run_command(ruff_cmd, allow_failure=True)
            if ruff_ret != 0:
                print(
                    "Warning: ruff reported issues that could not be fully fixed. "
                    "See output above for details.",
                    file=sys.stderr,
                )

            run_command(black_cmd, allow_failure=False)

    if args.check:
        print("ruff + black check complete.")
    else:
        print("ruff + black formatting complete.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
