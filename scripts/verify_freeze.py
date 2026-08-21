#!/usr/bin/env python3
"""Verify the working tree against the committed freeze manifest.

Reports any tracked file whose SHA-256 differs from the freeze, plus files added
or removed since. Exits non-zero on any difference, so it can gate a run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_manifest import tracked_files  # noqa: E402


def main() -> int:
    path = ROOT / "manifest" / "freeze.json"
    if not path.exists():
        print("no manifest/freeze.json; run scripts/build_manifest.py")
        return 2
    frozen = json.loads(path.read_text())["files"]
    current = tracked_files()

    changed = sorted(k for k in frozen.keys() & current.keys() if frozen[k] != current[k])
    removed = sorted(frozen.keys() - current.keys())
    added = sorted(current.keys() - frozen.keys())

    for label, items in (("CHANGED", changed), ("REMOVED", removed), ("ADDED", added)):
        for item in items:
            print(f"{label:8s} {item}")

    if changed or removed or added:
        print(f"\n{len(changed)} changed, {len(removed)} removed, {len(added)} added")
        return 1
    print(f"freeze verified: {len(frozen)} files match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
