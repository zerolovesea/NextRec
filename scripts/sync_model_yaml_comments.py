#!/usr/bin/env python3
"""Sync YAML model header comments from corresponding Python model modules.

This script updates only the *leading* comment block of each YAML file under
`nextrec_cli_preset/model_configs/`:
- Removes the YAML file's initial comment/blank prefix.
- Replaces it with the header comment extracted from the corresponding
  `nextrec/models/**/{model}.py` module.

Header extraction rule for Python module:
- Prefer the module docstring (top-level triple-quoted string).
- Otherwise, use consecutive `# ...` comment lines at the top of the file.

The extracted header is converted to YAML comment lines (`# ...`).

Usage:
  python scripts/sync_model_yaml_comments.py --dry-run
  python scripts/sync_model_yaml_comments.py --write

Defaults:
    - nextrec_cli_preset/model_configs
    - nextrec_studio/src/presets
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

_MODEL_LINE_RE = re.compile(r"^\s*model\s*:\s*([A-Za-z0-9_]+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class UpdateResult:
    yaml_path: Path
    model_name: str
    model_py_path: Path | None
    changed: bool
    reason: str


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "nextrec").exists():
            return parent
    return start.resolve()


def find_model_name(yaml_text: str, fallback_stem: str) -> str:
    match = _MODEL_LINE_RE.search(yaml_text)
    if match:
        return match.group(1)
    return fallback_stem


def index_first_yaml_content_line(lines: list[str]) -> int:
    """Return index of first non-blank, non-comment line."""
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "":
            continue
        if line.lstrip().startswith("#"):
            continue
        return idx
    return len(lines)


def strip_leading_shebang_and_encoding(lines: list[str], start_idx: int) -> int:
    idx = start_idx
    if idx < len(lines) and lines[idx].startswith("#!"):
        idx += 1
    # PEP 263 encoding comment: must be in first or second line (after shebang)
    if idx < len(lines) and lines[idx].lstrip().startswith("#") and "coding" in lines[idx]:
        idx += 1
    return idx


def extract_python_module_header(py_path: Path, max_lines: int = 200) -> list[str]:
    text = py_path.read_text(encoding="utf-8")
    raw_lines = text.splitlines()
    lines = raw_lines[:max_lines]

    idx = strip_leading_shebang_and_encoding(lines, 0)
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1

    if idx >= len(lines):
        return []

    line = lines[idx].lstrip()

    # Module docstring
    if line.startswith('"""') or line.startswith("'''"):
        quote = '"""' if line.startswith('"""') else "'''"
        opening = lines[idx]
        after_open = opening.split(quote, 1)[1]

        doc_lines: list[str] = []
        # Single-line docstring
        if quote in after_open:
            content = after_open.split(quote, 1)[0]
            doc_lines.append(content)
            return trim_blank_ends([l.rstrip("\r") for l in split_preserve_newlines(doc_lines)])

        # Multi-line docstring
        idx += 1
        while idx < len(lines):
            cur = lines[idx]
            if quote in cur:
                before_close = cur.split(quote, 1)[0]
                doc_lines.append(before_close)
                break
            doc_lines.append(cur)
            idx += 1

        return trim_blank_ends([l.rstrip("\r") for l in doc_lines])

    # Leading # comments
    comment_lines: list[str] = []
    while idx < len(lines):
        cur = lines[idx]
        if cur.strip() == "":
            if comment_lines:
                comment_lines.append("")
                idx += 1
                continue
            idx += 1
            continue
        if cur.lstrip().startswith("#"):
            payload = cur.lstrip()[1:]
            if payload.startswith(" "):
                payload = payload[1:]
            comment_lines.append(payload.rstrip("\r"))
            idx += 1
            continue
        break

    return trim_blank_ends(comment_lines)


def split_preserve_newlines(lines: list[str]) -> list[str]:
    """Split lines that may contain embedded newlines into true line list."""
    out: list[str] = []
    for line in lines:
        out.extend(line.splitlines())
    return out


def trim_blank_ends(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)
    while start < end and lines[start].strip() == "":
        start += 1
    while end > start and lines[end - 1].strip() == "":
        end -= 1
    return lines[start:end]


def format_as_yaml_comment_block(header_lines: list[str]) -> str:
    if not header_lines:
        return ""
    formatted: list[str] = []
    for line in header_lines:
        if line.strip() == "":
            formatted.append("#")
        else:
            formatted.append(f"# {line.rstrip()}" if not line.startswith("#") else f"# {line.lstrip('#').lstrip()}")
    return "\n".join(formatted) + "\n\n"


def find_model_py(models_dir: Path, model_name: str) -> Path | None:
    direct = list(models_dir.rglob(f"{model_name}.py"))
    if direct:
        return sorted(direct, key=lambda p: (len(p.parts), str(p)))[0]

    # case-insensitive fallback
    target = f"{model_name}.py".lower()
    for py in models_dir.rglob("*.py"):
        if py.name.lower() == target:
            return py
    return None


def update_one_yaml(yaml_path: Path, models_dir: Path) -> UpdateResult:
    original = yaml_path.read_text(encoding="utf-8")
    model_name = find_model_name(original, fallback_stem=yaml_path.stem)

    model_py = find_model_py(models_dir=models_dir, model_name=model_name)
    if model_py is None:
        # Remove existing leading comment block anyway (as requested), but cannot replace.
        lines = original.splitlines()
        first_content_idx = index_first_yaml_content_line(lines)
        new_text = "\n".join(lines[first_content_idx:]).lstrip("\n") + "\n"
        changed = new_text != (original if original.endswith("\n") else original + "\n")
        return UpdateResult(
            yaml_path=yaml_path,
            model_name=model_name,
            model_py_path=None,
            changed=changed,
            reason=(
                "model python file not found; removed YAML leading comments only"
                if changed
                else "model python file not found; no change"
            ),
        )

    header = extract_python_module_header(model_py)
    comment_block = format_as_yaml_comment_block(header)

    yaml_lines = original.splitlines()
    first_content_idx = index_first_yaml_content_line(yaml_lines)
    content_part = "\n".join(yaml_lines[first_content_idx:]).lstrip("\n")
    new_text = comment_block + content_part
    if not new_text.endswith("\n"):
        new_text += "\n"

    normalized_original = original if original.endswith("\n") else original + "\n"
    changed = new_text != normalized_original
    return UpdateResult(
        yaml_path=yaml_path,
        model_name=model_name,
        model_py_path=model_py,
        changed=changed,
        reason="updated" if changed else "no change",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=None,
        help="(Deprecated) Single YAML directory to process. Prefer --configs-dirs.",
    )
    parser.add_argument(
        "--configs-dirs",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "One or more YAML directories to process. If omitted, processes both "
            "nextrec_cli_preset/model_configs and nextrec_studio/src/presets (when present)."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Models directory (default: nextrec/models)",
    )
    parser.add_argument("--write", action="store_true", help="Write changes to files")
    parser.add_argument("--dry-run", action="store_true", help="Alias for not writing (default behavior)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__))
    models_dir: Path = args.models_dir or (repo_root / "nextrec" / "models")

    if args.configs_dir is not None and args.configs_dirs is not None:
        raise SystemExit("Use only one of --configs-dir or --configs-dirs")

    if args.configs_dirs is not None:
        configs_dirs = list(args.configs_dirs)
    elif args.configs_dir is not None:
        configs_dirs = [args.configs_dir]
    else:
        # Default: sync both CLI presets and Studio presets (if present)
        candidates = [
            repo_root / "nextrec_cli_preset" / "model_configs",
            repo_root / "nextrec_studio" / "src" / "presets",
        ]
        configs_dirs = [p for p in candidates if p.exists()]

    if not configs_dirs:
        raise SystemExit("No configs dirs found. Provide --configs-dirs explicitly.")

    for configs_dir in configs_dirs:
        if not configs_dir.exists():
            raise SystemExit(f"configs dir not found: {configs_dir}")
    if not models_dir.exists():
        raise SystemExit(f"models dir not found: {models_dir}")

    yaml_paths: list[Path] = []
    for configs_dir in configs_dirs:
        yaml_paths.extend(sorted(configs_dir.glob("*.yaml")))
    results: list[UpdateResult] = []

    for yaml_path in sorted(yaml_paths):
        res = update_one_yaml(yaml_path=yaml_path, models_dir=models_dir)
        results.append(res)

    changed = [r for r in results if r.changed]
    missing = [r for r in results if r.model_py_path is None]

    if args.verbose:
        for r in results:
            try:
                yaml_display = str(r.yaml_path.relative_to(repo_root))
            except ValueError:
                yaml_display = str(r.yaml_path)
            try:
                py_display = str(r.model_py_path.relative_to(repo_root)) if r.model_py_path else "<missing>"
            except ValueError:
                py_display = str(r.model_py_path) if r.model_py_path else "<missing>"
            status = "CHANGED" if r.changed else "OK"
            print(f"[{status}] {yaml_display} (model={r.model_name}, py={py_display}) - {r.reason}")

    print(f"YAML files: {len(results)}")
    print(f"Will change: {len(changed)}")
    if missing:
        print(f"Missing model py: {len(missing)}")

    if args.write:
        for r in changed:
            updated = update_one_yaml(yaml_path=r.yaml_path, models_dir=models_dir)
            # Recompute to get deterministic content (ensures no stale state)
            if not updated.changed:
                continue
            # Write
            original = r.yaml_path.read_text(encoding="utf-8")
            # Regenerate exact new text
            model_py = updated.model_py_path
            if model_py is None:
                lines = original.splitlines()
                first_content_idx = index_first_yaml_content_line(lines)
                new_text = "\n".join(lines[first_content_idx:]).lstrip("\n") + "\n"
            else:
                header = extract_python_module_header(model_py)
                comment_block = format_as_yaml_comment_block(header)
                yaml_lines = original.splitlines()
                first_content_idx = index_first_yaml_content_line(yaml_lines)
                content_part = "\n".join(yaml_lines[first_content_idx:]).lstrip("\n")
                new_text = comment_block + content_part
                if not new_text.endswith("\n"):
                    new_text += "\n"
            r.yaml_path.write_text(new_text, encoding="utf-8")

        print("Written changes.")
    else:
        print("Dry-run (no files written). Use --write to apply.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
