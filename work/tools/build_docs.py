#!/usr/bin/env python
"""Generate the two hand-off PDFs from the notebooks.

    python work/tools/build_docs.py

Both are generated, never hand-edited, so no figure can drift from the cell
that produced it. Regenerate after any notebook change.

    work/outputs/flyrank_the_work.pdf   what was done, found, and withdrawn
    work/outputs/flyrank_the_code.pdf   every code cell, explained
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_pdfs import (ASSIGNMENT, BODY, CODE, H1, H2, H3, OUT, ORDER, OUTP, para,
                        Paragraph, PageBreak, Preformatted, SMALL, Spacer,
                        build, cell_out, cell_src, colors, esc, load_nb,
                        md_to_flowables, mm)
from pdf_explanations import EXPLAIN

NB = {Path(p).stem: p for p in __import__("glob").glob(
    str(Path(__file__).resolve().parents[2] / "work" / "notebooks" / "w0*.ipynb"))}


def cover(title: str, subtitle: str, blurb: str) -> list:
    return [Spacer(1, 40 * mm),
            Paragraph(title, H1),
            Paragraph(subtitle, ParagraphStyleSub()),
            Spacer(1, 6 * mm),
            Paragraph(blurb, BODY),
            Spacer(1, 10 * mm),
            Paragraph("Generated from the notebooks by "
                      "<font face='Courier'>work/tools/build_docs.py</font>. "
                      "Every figure below is printed by a cell that ran; none is "
                      "transcribed by hand.", SMALL),
            PageBreak()]


def ParagraphStyleSub():
    from reportlab.lib.styles import ParagraphStyle
    return ParagraphStyle("SUB", parent=BODY, fontSize=12, leading=16,
                          textColor=colors.HexColor("#555555"))


# --------------------------------------------------------------------- the work
def build_work():
    flow = cover(
        "The Work",
        "FlyRank ML internship &mdash; ML-02 to ML-09",
        "A walkthrough of every assignment: what it set out to do, what it found, and what it "
        "withdrew. The written sections of each notebook are reproduced in order, so the reasoning "
        "path is visible rather than only its conclusions. Where a finding was later corrected, "
        "the correction appears in place rather than replacing the original.")

    flow.append(para("How to read this", H2))
    flow += md_to_flowables(
        "This project spent most of its effort discovering that its own measurements were "
        "constructed wrongly, and correcting them. That is why several sections read as "
        "withdrawals rather than results.\n\n"
        "Three ideas recur and are worth holding while reading:\n\n"
        "**The shared-denominator artefact.** If a feature and the target are both built from the "
        "same window, a relationship appears between them before any page behaviour enters. This "
        "cost three separate findings.\n\n"
        "**The null simulation.** Replace the real future with a random window from a page's own "
        "history. The features, the level and the volatility are unchanged; only the information "
        "about what happened next is destroyed. Whatever survives is arithmetic.\n\n"
        "**Base rates beside every score.** A precision of 0.94 on a pool that declines at 0.80 is "
        "not a strong result. Two findings in this project were withdrawn for missing this.")
    flow.append(PageBreak())

    for stem in ORDER:
        if stem not in NB:
            continue
        code, title = ASSIGNMENT[stem]
        nb = load_nb(NB[stem])
        mds = [c for c in nb["cells"] if c["cell_type"] == "markdown"
               and cell_src(c).strip()]
        if not mds:
            continue
        flow.append(para(f"{code} &mdash; {title}", H1))
        flow.append(para(f"Source notebook: "
                              f"<font face='Courier'>{Path(NB[stem]).name}</font> "
                              f"&nbsp;&middot;&nbsp; {len(mds)} written sections, "
                              f"{sum(1 for c in nb['cells'] if c['cell_type'] == 'code' and cell_src(c).strip())} "
                              f"code cells", SMALL))
        flow.append(Spacer(1, 4 * mm))
        for c in mds:
            src = cell_src(c)
            if src.lstrip().startswith("<a href") or "Open In Colab" in src:
                continue
            flow += md_to_flowables(src)
        flow.append(PageBreak())

    build(OUT / "flyrank_the_work.pdf", "FlyRank ML internship - The Work", flow)


# --------------------------------------------------------------------- the code
def summarise(src: str) -> str:
    import re
    bits = []
    defs = re.findall(r"^def (\w+)", src, re.M)
    if defs:
        bits.append("defines " + ", ".join(f"<font face='Courier'>{d}</font>" for d in defs))
    if "con.sql" in src or "read_parquet" in src:
        bits.append("queries the warehouse")
    if "GroupShuffleSplit" in src:
        bits.append("evaluates on client-grouped splits")
    if "print(" in src:
        bits.append("prints results")
    return ("NO AUTHORED EXPLANATION. Generated summary: this cell "
            + ("; ".join(bits) if bits else "runs setup code") + ".")


def build_code():
    flow = cover(
        "The Code",
        "Every code cell, explained",
        "One entry per code cell, in notebook order: what the cell does and why it is there, "
        "followed by the code itself and the output it produced on the last run. Cells that only "
        "hold the skeleton's placeholder comment are listed and skipped.")

    total = 0
    missing = []
    for stem in ORDER:
        if stem not in NB:
            continue
        code_id, title = ASSIGNMENT[stem]
        nb = load_nb(NB[stem])
        cells = [(i, c) for i, c in enumerate(nb["cells"])
                 if c["cell_type"] == "code" and cell_src(c).strip()]
        real = [(i, c) for i, c in cells
                if not cell_src(c).strip().startswith("# This cell is for CODE")]
        flow.append(para(f"{code_id} &mdash; {title}", H1))
        flow.append(para(f"<font face='Courier'>{Path(NB[stem]).name}</font> "
                              f"&nbsp;&middot;&nbsp; {len(real)} code cells "
                              f"({len(cells) - len(real)} empty skeleton stubs skipped)", SMALL))
        flow.append(Spacer(1, 4 * mm))

        for idx, c in real:
            total += 1
            src = cell_src(c).rstrip()
            note = EXPLAIN.get((stem, idx))
            if note is None:
                missing.append((stem, idx, src.splitlines()[0][:55]))
                note = summarise(src)
            flow.append(para(f"cell {idx}", H3))
            flow.append(para(esc(note).replace("&lt;font face=&#39;Courier&#39;&gt;", "")
                                  if "NO AUTHORED" not in note else
                                  f"<font color='#b00020'>{esc(note)}</font>", BODY))
            flow.append(Preformatted(src, CODE, maxLineLength=132))
            out = cell_out(c)
            if out:
                flow.append(Spacer(1, 1.5 * mm))
                flow.append(para("output", SMALL))
                flow.append(Preformatted(out, OUTP, maxLineLength=140))
            flow.append(Spacer(1, 5 * mm))
        flow.append(PageBreak())

    if missing:
        print(f"WARNING: {len(missing)} cells have no authored explanation. Inserting a cell "
              f"shifts every index after it, so this is usually key drift rather than a new cell:")
        for stem, idx, head in missing:
            print(f"    ({stem!r}, {idx}) -> {head}")
    print(f"code document covers {total} cells")
    build(OUT / "flyrank_the_code.pdf", "FlyRank ML internship - The Code", flow)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build_work()
    build_code()
