#!/usr/bin/env python
"""Every number written in prose must come from a cell that actually ran.

Reads the markdown cells of each notebook and the body of the capstone report,
pulls out every numeric token, and checks each one against the pooled stdout of
every executed code cell. A number that appears nowhere in any output is either
carried in from a scratch script, stale from an earlier revision, or mistyped --
all three have happened in this project, and none survive a re-read.

    python work/tools/check_claims.py            # report and exit 1 on failure
    python work/tools/check_claims.py --list     # show what each number matched

Matching is deliberately generous, because prose legitimately reformats numbers:

    literal          46.2         -> "46.2"
    thousands        134,399      -> "134399"
    percent/decimal  3.76%        -> "0.0376"
    rounding         52.9         -> "52.88"
    external         0.740        -> found in docs/

Anything still unmatched is either a real defect or belongs in ALLOWED below,
with a reason. Silence is not an option -- an exception has to be written down.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

NOTEBOOKS = sorted(glob.glob(str(ROOT / "work" / "notebooks" / "*.ipynb")))
PROSE_FILES = [ROOT / "work" / "capstone_report.md"]
DOC_GLOBS = ["docs/**/*.md", "README.md", "GUIDE.md"]

# Numbers that are deliberately not backed by a cell, each with a reason.
# Retracted figures belong here: a correction has to name what it corrects,
# but the superseded value no longer appears in any output.
ALLOWED = {
    "12917991": "Google Search Console docs URL, not a measurement",
    "48.5": "retracted per-client recall; named in the correction that replaces it with 3.76%",
    "44.8": "retracted Sparrow ceiling; named in the correction that replaces it with 4.2%",
    "64.7": "retracted Eagle ceiling; named in the correction that replaces it with 17.0%",
    "1.72": "retracted single-split lift; named in the correction that withdraws it",
    "47,593": "retracted dense-page count; named in ML-06's revision log against the real 70,013",
    "0.044": "retracted Spearman; named in ML-06's revision log against -0.0665 / -0.1505",
    "0.89": "retracted autocorrelation; named in the revision logs against 0.799",
    "94,550": "retracted log-ratio range; named in the revision log",
    "0.819": "pre-pooling per-seed recall; no longer produced, named only in the correction",
    "39,131": "flag-test cell row, printed as part of a wider table",
    "0.169": "retracted split-inflation figure; named in ML-05's correction table beside the measured +0.091",
    "0.934": "retracted collinearity figure; named beside the measured r = 0.997",
    "0.234": "retracted CTR rank correlation; named beside the measured -0.288",
    "0.217": "retracted CTR rank correlation; named beside the measured -0.231",
    "0.239": "retracted CTR rank correlation; named beside the measured -0.231",
    "28.2": "recovery rate after the 30-day baseline fix; the superseded 90-day pipeline is not rerun",
    "18.3": "recovery rate before the 30-day baseline fix; same reason",
    "49.62": "label rate before the off-by-one fix; the 31-day window is not rerun",
    "50.95": "label rate after the off-by-one fix, measured under the superseded binary label",
    "0.080": "Pearson position/CTR from notebooks/01_first_look_and_discovery.ipynb, outside work/",
}

TOKEN_RE = re.compile(
    r"\d[\d,]*\.\d+%?"        # 46.2   3.76%   1,234.5
    r"|\d[\d,]*%"             # 22.3%  100%
    r"|\d{1,3}(?:,\d{3})+"    # 47,593  1,339  134,399   (any digits before the comma)
    r"|\d{3,}"                # 70013
)
NUM_IN_OUTPUT_RE = re.compile(r"\d+\.?\d*")

# prose constructs that carry digits which are not claims
STRIP_PATTERNS = [
    re.compile(r"\]\([^)]*\)"),          # markdown link targets
    re.compile(r"https?://\S+"),          # bare URLs
    re.compile(r"```.*?```", re.S),       # fenced code
    re.compile(r"`[^`\n]*`"),             # inline code -- formulas, column names
]


def strip_noise(text: str) -> str:
    for pat in STRIP_PATTERNS:
        text = pat.sub(" ", text)
    return text


def load_pool() -> tuple[str, set[float]]:
    """Concatenated stdout of every executed code cell, plus its numbers."""
    chunks: list[str] = []
    for path in NOTEBOOKS:
        nb = json.loads(Path(path).read_text(encoding="utf-8"))
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            for out in cell.get("outputs", []):
                text = out.get("text")
                if isinstance(text, list):
                    text = "".join(text)
                if text:
                    chunks.append(text)
                data = out.get("data", {}).get("text/plain")
                if isinstance(data, list):
                    chunks.append("".join(data))
                elif isinstance(data, str):
                    chunks.append(data)
    pool = "\n".join(chunks)
    values = set()
    for m in NUM_IN_OUTPUT_RE.finditer(pool.replace(",", "")):
        try:
            values.add(float(m.group()))
        except ValueError:
            pass
    return pool, values


def load_docs() -> str:
    parts = []
    for pattern in DOC_GLOBS:
        for path in ROOT.glob(pattern):
            if path.is_file():
                parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def explain(token: str, pool: str, values: set[float], docs: str) -> str | None:
    """Return how the token is accounted for, or None if it is not."""
    bare = token.rstrip("%")
    flat = bare.replace(",", "")

    if bare in ALLOWED or flat in ALLOWED:
        return f"allowed: {ALLOWED.get(bare) or ALLOWED[flat]}"
    if bare in pool:
        return "literal"
    if flat in pool.replace(",", ""):
        return "thousands separator"

    try:
        value = float(flat)
    except ValueError:
        return "not numeric"

    if value in values:
        return "exact value"

    if token.endswith("%"):
        frac = value / 100
        for digits in (3, 4, 5, 6):
            if f"{frac:.{digits}f}" in pool:
                return f"percent of {frac:.{digits}f}"
        if any(abs(v - frac) < 10 ** -6 for v in values):
            return "percent of a printed decimal"

    decimals = len(bare.split(".")[1]) if "." in bare else 0
    for other in values:
        if round(other, decimals) == value:
            return f"rounded from {other:g}"
        if decimals == 0 and abs(other - value) < 0.5:
            return f"rounded from {other:g}"

    if bare in docs or flat in docs.replace(",", ""):
        return "external: docs/"

    return None


def scan() -> list[tuple[str, str, str]]:
    """Return (source, token, context) for every unaccounted number."""
    pool, values = load_pool()
    docs = load_docs()
    misses: list[tuple[str, str, str]] = []
    matched = 0

    units: list[tuple[str, str]] = []
    for path in NOTEBOOKS:
        nb = json.loads(Path(path).read_text(encoding="utf-8"))
        name = Path(path).name
        for i, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "markdown":
                continue
            src = cell.get("source")
            src = "".join(src) if isinstance(src, list) else (src or "")
            units.append((f"{name} cell {i}", src))
    for path in PROSE_FILES:
        if path.exists():
            units.append((path.name, path.read_text(encoding="utf-8")))

    seen: set[tuple[str, str]] = set()
    for where, text in units:
        for token in TOKEN_RE.findall(strip_noise(text)):
            if len(token.rstrip("%")) < 3:
                continue
            key = (where, token)
            if key in seen:
                continue
            seen.add(key)
            reason = explain(token, pool, values, docs)
            if reason is None:
                idx = text.find(token)
                context = " ".join(text[max(0, idx - 70): idx + 40].split())
                misses.append((where, token, context))
            else:
                matched += 1

    print(f"scanned {len(units)} prose units | {matched} numbers accounted for | "
          f"{len(misses)} unaccounted")
    return misses


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true",
                    help="print how every number was matched, not just the failures")
    args = ap.parse_args()

    if args.list:
        pool, values = load_pool()
        docs = load_docs()
        for path in NOTEBOOKS:
            nb = json.loads(Path(path).read_text(encoding="utf-8"))
            for i, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") != "markdown":
                    continue
                src = cell.get("source")
                src = "".join(src) if isinstance(src, list) else (src or "")
                for token in sorted(set(TOKEN_RE.findall(strip_noise(src)))):
                    if len(token.rstrip("%")) < 3:
                        continue
                    print(f"  {Path(path).name} c{i:<3} {token:>12}  "
                          f"{explain(token, pool, values, docs) or 'UNACCOUNTED'}")
        return 0

    misses = scan()
    if not misses:
        print("\nOK - every number in prose traces to a cell that ran.")
        return 0

    print("\nUnaccounted numbers - each is stale, mistyped, or from a scratch script:\n")
    for where, token, context in misses:
        print(f"  {where}")
        print(f"      {token}   ...{context}...")
    print("\nFix the number, compute it in the notebook, or add it to ALLOWED with a reason.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
