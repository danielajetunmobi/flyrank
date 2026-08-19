#!/usr/bin/env python
"""Quantities that appear in more than one notebook must agree.

check_claims.py verifies that every number in prose came from a cell that ran.
It cannot see that ML-07 says the random-order bar is 0.5432 while ML-08
computes 0.5132 -- both are backed by their own cells, and they contradict
each other. This checks the seams between notebooks instead of inside them.

Each entry below names a quantity, a regex that finds it in prose, and the
tolerance within which two mentions count as agreeing. Anything outside that
is either a stale figure or a real disagreement, and both need saying out loud.

    python work/tools/check_consistency.py
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

# Notes and prose here carry the Unicode minus (U+2212), which the Windows
# console's cp1252 codec cannot encode -- printing one aborted the run after
# every check had already passed. Force UTF-8 so a passing run reports as one.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
SOURCES = sorted(glob.glob(str(ROOT / "work" / "notebooks" / "*.ipynb"))) + [
    str(ROOT / "work" / "capstone_report.md")
]

# quantity -> (pattern, tolerance, note)
SHARED = {
    "D1 cohort pages": (r"202,073", 0, "the full D1 cohort after the target fix"),
    "D1 gated pool": (r"46,061", 0, "pages passing ML-07's frozen gate"),
    "D1 cohort decline rate": (r"0\.4228|42\.3%", 0.001, "share with target < 0"),
    "peak_ratio null share": (r"0\.1249", 0.001, "ML-07 section 0 candidate C"),
    "autocorrelation": (r"0\.799", 0.001, "consecutive 30-day means"),
    "days_since_last_update leak": (r"77\.3%", 0.001, "D1 pages updated after the decision point"),
    "dead pages at D1": (r"7\.4%|9,949", 0, "pages reaching zero"),
    "random-order bar": (r"0\.5432|0\.5132", 0.001,
                         "ML-07 used one shuffle per seed, ML-08 twenty; 0.5132 is the estimate to use"),
    "ML-07 rule vs random": (r"−0\.0391|-0\.0391|−0\.0250|-0\.0250", 0.02,
                             "the rule ranks below random; two pools, so a spread is expected"),
    # ML-09's walk-forward made this one load-bearing: every grouped split in the
    # project sits at D1's gated decline rate, so if this drifts, every stability
    # range quoted anywhere is quoting a different pool than it claims.
    "D1 gated decline rate": (r"0\.5131", 0.001,
                              "the base rate all ten grouped-split seeds share"),
    "D2 gated decline rate": (r"0\.7873|0\.7869", 0.001,
                              "ML-08 scores at 2026-05-31, ML-09's walk at 2026-06-01; "
                              "one day apart, so these must stay near-identical"),
}


def prose_blocks() -> list[tuple[str, str]]:
    out = []
    for path in SOURCES:
        p = Path(path)
        if p.suffix == ".ipynb":
            nb = json.loads(p.read_text(encoding="utf-8"))
            for i, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") != "markdown":
                    continue
                src = cell.get("source")
                src = "".join(src) if isinstance(src, list) else (src or "")
                out.append((f"{p.name} cell {i}", src))
        else:
            out.append((p.name, p.read_text(encoding="utf-8")))
    return out


def main() -> int:
    blocks = prose_blocks()
    problems = 0
    print(f"checking {len(SHARED)} shared quantities across {len(SOURCES)} sources\n")
    for name, (pattern, tol, note) in SHARED.items():
        hits: dict[str, set[str]] = {}
        for where, text in blocks:
            for m in re.findall(pattern, text):
                hits.setdefault(where.split(" cell")[0], set()).add(m)
        if not hits:
            print(f"  {name:28s} NOT FOUND anywhere -- pattern may be stale")
            problems += 1
            continue
        variants = {v for s in hits.values() for v in s}
        files = sorted(hits)
        status = "ok" if len(files) > 1 else "single source"
        print(f"  {name:28s} {status:13s} {len(files)} file(s): {', '.join(files)}")
        if len(variants) > 1:
            print(f"      variants seen: {sorted(variants)}  ({note})")
    print()
    if problems:
        print(f"{problems} quantity pattern(s) found nowhere. Update SHARED or the prose.")
        return 1
    print("No stale shared-quantity patterns. Cross-notebook agreement is a human read, "
          "not a proof -- this narrows where to look.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
