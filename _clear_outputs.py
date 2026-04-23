"""Strip oversized outputs from notebooks so they fit under GitHub file size limits.

Usage:
    python _clear_outputs.py notebooks/*.ipynb [--max-bytes 2000000]

Default policy:
    * Drop any cell output whose serialized payload exceeds ``--max-bytes``.
    * Leave smaller text/plain, image/png, JSON outputs intact.
    * Add a placeholder output so the cell isn't blank.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _output_size(output: dict) -> int:
    try:
        return len(json.dumps(output, ensure_ascii=False))
    except Exception:
        return 0


def _strip_oversized(nb: dict, max_bytes: int) -> tuple[int, int]:
    stripped = 0
    kept = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs", [])
        new_outputs = []
        cell_stripped = False
        for out in outputs:
            size = _output_size(out)
            if size > max_bytes:
                cell_stripped = True
                stripped += 1
                continue
            new_outputs.append(out)
            kept += 1
        if cell_stripped:
            new_outputs.append(
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": [
                        f"[output stripped: exceeded {max_bytes:,} bytes — "
                        "see data/outputs/*.html for full interactive map]\n"
                    ],
                }
            )
        cell["outputs"] = new_outputs
    return stripped, kept


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    max_bytes = 2_000_000
    for a in sys.argv[1:]:
        if a.startswith("--max-bytes"):
            _, _, val = a.partition("=")
            if val:
                max_bytes = int(val)
    for path in args:
        p = Path(path)
        nb = json.loads(p.read_text(encoding="utf-8"))
        stripped, kept = _strip_oversized(nb, max_bytes)
        p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        size_mb = p.stat().st_size / 1_048_576
        print(f"{p}: stripped={stripped} kept={kept} final={size_mb:.2f} MB")


if __name__ == "__main__":
    main()
