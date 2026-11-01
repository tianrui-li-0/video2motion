#!/usr/bin/env python3
"""Fail if a public checkout contains forbidden assets or private paths."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".ckpt", ".onnx", ".pth", ".pt", ".pkl", ".blend", ".glb"}
FORBIDDEN_TEXT = [
    re.compile(r"[A-Za-z]:\\Users\\"),
    re.compile(r"/mnt/[a-z]/RobotProjects", re.IGNORECASE),
    re.compile(r"yolox_humanart\.onnx", re.IGNORECASE),
    re.compile(r"SMPLX_NEUTRAL", re.IGNORECASE),
]
ALLOWED_TEXT = {
    ROOT / "README.md",
    ROOT / "THIRD_PARTY.md",
    ROOT / "scripts" / "run_gemx_manual_boxes.py",
    Path(__file__).resolve(),
}


def main() -> None:
    problems = []
    for path in ROOT.rglob("*"):
        relative_parts = path.relative_to(ROOT).parts
        if (
            not path.is_file()
            or any(part in {".git", ".venv", ".pytest_cache", "__pycache__", "build", "dist"} for part in relative_parts)
            or any(part.endswith(".egg-info") for part in relative_parts)
        ):
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden binary: {path.relative_to(ROOT)}")
        if path.stat().st_size > 25 * 1024 * 1024:
            problems.append(f"file larger than 25 MiB: {path.relative_to(ROOT)}")
        if path.suffix.lower() in {".py", ".md", ".toml", ".yaml", ".yml", ".json", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in FORBIDDEN_TEXT:
                if pattern.search(text) and path not in ALLOWED_TEXT:
                    problems.append(
                        f"forbidden text {pattern.pattern!r}: {path.relative_to(ROOT)}"
                    )
    if problems:
        raise SystemExit("Release audit failed:\n- " + "\n- ".join(problems))
    print("Release audit passed")


if __name__ == "__main__":
    main()
