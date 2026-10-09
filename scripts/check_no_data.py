#!/usr/bin/env python3
"""Fail if the repository tracks data that the ADNI data use agreement forbids sharing.

Checks every file tracked by git (or every file under the repository root
when git is unavailable) for:

* imaging and array formats (.nii, .nii.gz, .dcm, .npy, .npz, .mgz, .img/.hdr);
* model checkpoints (.pt, .pth, .ckpt);
* real ADNI subject IDs (``NNN_S_NNNN`` with a site other than 000, which is
  reserved for synthetic test data);
* any single file larger than 5 MB.

Exit status 1 lists the offending files. Run locally with
``python scripts/check_no_data.py`` or through the pre-commit hook.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_SUFFIXES = (
    ".nii", ".nii.gz", ".dcm", ".npy", ".npz", ".mgz", ".img", ".hdr", ".pt", ".pth", ".ckpt",
)  # fmt: skip
REAL_ID = re.compile(r"(?<!\d)(?!000_)\d{3}_S_\d{4}(?!\d)")
MAX_BYTES = 5 * 1024 * 1024
SELF = Path(__file__).resolve()


def tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
        ).stdout
        return [ROOT / p for p in out.decode().split("\0") if p]
    except (OSError, subprocess.CalledProcessError):
        return [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]


def main() -> int:
    problems = []
    for f in tracked_files():
        if not f.exists():
            continue
        rel = f.relative_to(ROOT)
        name = f.name.lower()
        if name.endswith(BLOCKED_SUFFIXES):
            problems.append(f"{rel}: data or checkpoint file type")
            continue
        if f.stat().st_size > MAX_BYTES:
            problems.append(f"{rel}: larger than 5 MB")
            continue
        if f.resolve() == SELF:
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits = sorted(set(REAL_ID.findall(text)))
        if hits:
            problems.append(f"{rel}: contains {len(hits)} ADNI-style subject ID(s)")
    if problems:
        print("Data guard failed:")
        print("\n".join(f"  {p}" for p in problems))
        return 1
    print("Data guard passed: no imaging data, checkpoints or subject IDs are tracked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
