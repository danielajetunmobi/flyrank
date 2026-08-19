#!/usr/bin/env python
"""Build the two hand-off PDFs from the notebooks.

    python work/tools/build_pdfs.py

Both documents are generated, never hand-edited, so a number in a PDF cannot
drift from the cell that produced it. Regenerate after any notebook change.

    work/outputs/flyrank_the_work.pdf   what was done, found, and retracted
    work/outputs/flyrank_the_code.pdf   every code cell, explained

Explanations for the code document live in EXPLAIN below, keyed by notebook
stem and cell index. A cell with no entry is rendered with a generated
summary and flagged, so nothing is silently skipped.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "outputs"
NOTEBOOKS = sorted(glob.glob(str(ROOT / "work" / "notebooks" / "w0*.ipynb")))

ASSIGNMENT = {
    "w01_research_question": ("ML-02", "Research question and provisional lane"),
    "w02_ml_task_framing": ("ML-03", "Framing the lane as an ML task"),
    "w03_data_contract": ("ML-04", "The data contract"),
    "w03_feature_leakage_check": ("ML-05", "Feature vector and leakage probe"),
    "w04_signal_audit": ("ML-06", "Signal audit"),
    "w04_baseline_score": ("ML-07", "Baseline action score"),
    "w05_model": ("ML-08", "Model training and comparison"),
    "w06_validation_audit": ("ML-09", "Validation and claim audit"),
    "w07_action_playbook": ("ML-10", "Action playbook (not started)"),
}
ORDER = ["w01_research_question", "w02_ml_task_framing", "w03_data_contract",
         "w03_feature_leakage_check", "w04_signal_audit", "w04_baseline_score",
         "w05_model", "w06_validation_audit", "w07_action_playbook"]

# --------------------------------------------------------------------------- styles
SS = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=SS["Heading1"], fontSize=20, leading=24,
                    spaceAfter=8, textColor=colors.HexColor("#111111"))
H2 = ParagraphStyle("H2", parent=SS["Heading2"], fontSize=14, leading=18,
                    spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#1a1a1a"))
H3 = ParagraphStyle("H3", parent=SS["Heading3"], fontSize=11, leading=15,
                    spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#333333"))
BODY = ParagraphStyle("BODY", parent=SS["BodyText"], fontSize=9.5, leading=14,
                      alignment=TA_LEFT, spaceAfter=6)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8, leading=11,
                       textColor=colors.HexColor("#555555"))
CODE = ParagraphStyle("CODE", parent=SS["Code"], fontSize=6.6, leading=8.2,
                      textColor=colors.HexColor("#222222"))
OUTP = ParagraphStyle("OUTP", parent=SS["Code"], fontSize=6.4, leading=8,
                      textColor=colors.HexColor("#0b4f6c"))
LIST = ParagraphStyle("LIST", parent=BODY, leftIndent=5 * mm, spaceAfter=2)


def esc(t: str) -> str:
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")


def md_to_flowables(src: str, max_chars: int = 100_000) -> list:
    """Render a markdown cell as flowables. Tables become real tables."""
    out, buf, rows = [], [], []
    listing = False

    def flush():
        nonlocal buf, listing
        if buf:
            out.append(para(_inline(" ".join(buf)), LIST if listing else BODY))
            buf = []

    for raw in src[:max_chars].split("\n"):
        line = raw.rstrip()
        if line.startswith("|") and line.count("|") >= 2:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):
                rows.append(cells)
            continue
        if rows:
            flush(); listing = False
            out.append(_table(rows)); rows = []
        if not line.strip():
            flush(); listing = False
            continue
        # A horizontal rule. Without this it is not a heading, a table or blank,
        # so it lands in buf and prints as a literal "---" paragraph. The
        # notebooks rarely use one; the capstone report uses ten.
        if re.fullmatch(r"[-*_]{3,}", line.strip()):
            flush(); listing = False
            out.append(Spacer(1, 3 * mm))
            continue
        if line.startswith("#"):
            flush(); listing = False
            level = len(line) - len(line.lstrip("#"))
            out.append(para(_inline(line.lstrip("# ").strip()),
                                 H2 if level <= 2 else H3))
            continue
        if line.startswith(">"):
            line = line.lstrip("> ").strip()
        # A list item ends the previous paragraph and starts an indented one.
        # Without this every bullet in a block runs into its neighbours as one
        # paragraph with stray "-" characters mid-sentence.
        m = _LIST_RE.match(line)
        if m:
            flush()
            listing = True
            # Keep an ordered list's numbers: the capstone's reproducibility
            # gaps are referred to elsewhere as "gap 3" and "gap 4".
            marker = m.group(0).strip()
            line = ("•" if marker in "-*+" else marker) + " " + line[m.end():]
        buf.append(line)
    if rows:
        out.append(_table(rows))
    flush()
    return out


def _inline(t: str) -> str:
    """Markdown to reportlab markup.

    Code spans are lifted out before the emphasis passes run. A literal asterisk
    inside backticks -- `has_*` is the one that broke this -- otherwise opens an
    <i> that never closes, and reportlab takes the whole document down with it.
    """
    spans: list[str] = []

    def _stash(m):
        spans.append(m.group(1))
        return f"\x00{len(spans) - 1}\x00"

    t = re.sub(r"`([^`]+)`", _stash, t)
    t = esc(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])", r"<i>\1</i>", t)
    for i, code in enumerate(spans):
        t = t.replace(f"\x00{i}\x00", f'<font face="Courier">{esc(code)}</font>')
    return t


def para(text: str, style) -> Paragraph:
    """A Paragraph that degrades to plain text rather than killing the build."""
    try:
        return Paragraph(text, style)
    except Exception:
        return Paragraph(esc(re.sub(r"<[^>]+>", "", text)), style)


def _table(rows: list[list[str]]) -> Table:
    width = 170 * mm
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    data = [[para(_inline(c), SMALL) for c in r] for r in rows]
    t = Table(data, colWidths=[width / ncol] * ncol, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def load_nb(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cell_src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else (s or "")


def cell_out(cell: dict, limit: int = 2600) -> str:
    parts = []
    for o in cell.get("outputs", []):
        t = o.get("text")
        if isinstance(t, list):
            t = "".join(t)
        if t:
            parts.append(t)
        d = o.get("data", {}).get("text/plain")
        if isinstance(d, list):
            parts.append("".join(d))
        elif isinstance(d, str):
            parts.append(d)
    text = "".join(parts)
    text = "\n".join(l for l in text.split("\n")
                     if "pip install" not in l and "notice]" not in l
                     and "Note: you may need" not in l)
    return text[:limit].rstrip()


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(20 * mm, 12 * mm, doc.title)
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"page {canvas.getPageNumber()}")
    canvas.restoreState()


def build(path: Path, title: str, flowables: list):
    doc = SimpleDocTemplate(str(path), pagesize=A4, title=title,
                            author="FlyRank ML internship",
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=20 * mm)
    doc.title = title
    doc.build(flowables, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {path}  ({path.stat().st_size / 1024:.0f} KB)")
