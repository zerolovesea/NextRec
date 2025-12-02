#!/usr/bin/env python3
"""Utility to run all tutorial Python scripts sequentially."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TUTORIAL_DIR.parent
EXCLUDED_DIRS = {"__pycache__", "notebooks"}
EXCLUDED_FILES = {Path(__file__).name}


def discover_scripts() -> list[Path]:
    """Return tutorial scripts directly under the tutorials/ directory."""
    scripts: list[Path] = []
    for path in TUTORIAL_DIR.glob("*.py"):
        if path.name in EXCLUDED_FILES:
            continue
        scripts.append(Path(path.name))
    scripts.sort()
    return scripts


def run_script(script: Path) -> int:
    """Run a single tutorial script and return its exit code."""
    abs_path = TUTORIAL_DIR / script
    print(f"\n=== Running tutorial: {script} ===")
    result = subprocess.run([sys.executable, str(abs_path)], cwd=REPO_ROOT, check=False)
    if result.returncode == 0:
        print(f"=== Completed: {script} ===\n")
    else:
        print(f"!!! Failed: {script} (exit code {result.returncode})")
    return result.returncode


def main() -> int:
    scripts = discover_scripts()
    if not scripts:
        print("No tutorial scripts found.")
        return 0

    print("Tutorial scripts to run:")
    for script in scripts:
        print(f"  - {script}")

    for script in scripts:
        exit_code = run_script(script)
        if exit_code != 0:
            return exit_code

    print("All tutorial scripts completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
