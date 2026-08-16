"""
STEP 1 — Diagnostic. Run this FIRST. It writes nothing to the pipeline.

It tells you exactly how the ACG PDF is typeset: which font sizes exist, which
one is body text, and every line that is a plausible heading. You need this
before any parsing logic, because every failed run so far came from guessing
what the headings look like instead of looking.

Run:   python 1_inspect.py
Then:  open data/processed/inspect.txt  and paste it back.
"""
import glob
import os
import sys
from collections import Counter

import pymupdf

# The Windows console defaults to cp1252 and cannot print the "fi" ligature
# that lives in this PDF. Force UTF-8 on stdout, never let printing crash a run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT = "data/processed/inspect.txt"


def find_pdf():
    """Auto-locate the guideline PDF so a wrong path can't fail the run."""
    for pat in ("data/raw/*.pdf", "data/*.pdf", "*.pdf", "**/*.pdf"):
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    raise SystemExit("No PDF found. Put the ACG PDF in data/raw/ and re-run.")


PDF_PATH = find_pdf()


def lines_with_style(doc):
    out = []
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        for block in page.get_text("dict", sort=True)["blocks"]:
            if block.get("type") != 0:
                continue
            for ln in block["lines"]:
                spans = [s for s in ln["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans)
                sizes, bold, total = Counter(), 0, 0
                fonts = Counter()
                for s in spans:
                    n = len(s["text"])
                    sizes[round(s["size"], 1)] += n
                    fonts[s["font"]] += n
                    if (s["flags"] & 16) or "bold" in s["font"].lower():
                        bold += n
                    total += n
                out.append({
                    "page": pno + 1,
                    "text": text.strip(),
                    "size": sizes.most_common(1)[0][0],
                    "font": fonts.most_common(1)[0][0],
                    "bold": bold / max(total, 1) > 0.55,
                    "x0": round(ln["bbox"][0], 1),
                })
    return out


def main():
    print("PDF:", PDF_PATH)
    doc = pymupdf.open(PDF_PATH)
    lines = lines_with_style(doc)

    by_size = Counter()
    for l in lines:
        by_size[l["size"]] += len(l["text"])
    body = by_size.most_common(1)[0][0]

    rep = []
    rep.append(f"PDF: {PDF_PATH}")
    rep.append(f"PAGES: {doc.page_count}")
    rep.append(f"TOTAL LINES: {len(lines)}")
    rep.append(f"BODY FONT SIZE (most chars): {body}")
    rep.append("")
    rep.append("=== FONT SIZE HISTOGRAM (size -> chars, lines) ===")
    line_ct = Counter(l["size"] for l in lines)
    for size, chars in sorted(by_size.items(), key=lambda x: -x[0]):
        tag = "  <-- BODY" if size == body else ""
        rep.append(f"  size {size:>5}  chars {chars:>7}  lines {line_ct[size]:>5}{tag}")

    rep.append("")
    rep.append("=== FONTS IN USE ===")
    for f, c in Counter(l["font"] for l in lines).most_common():
        rep.append(f"  {f:<40} {c}")

    rep.append("")
    rep.append("=== HEADING CANDIDATES (size > body, or bold, or ALL CAPS) ===")
    rep.append("    p    size  B  x0     text")
    for l in lines:
        t = l["text"]
        if not t or len(t.split()) > 16:
            continue
        letters = [c for c in t if c.isalpha()]
        allcaps = bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.75
        if l["size"] > body * 1.02 or l["bold"] or allcaps:
            b = "B" if l["bold"] else " "
            rep.append(f"  {l['page']:>3}  {l['size']:>5}  {b}  {l['x0']:>5}   {t[:90]}")

    rep.append("")
    rep.append("=== FIRST 3 LINES OF EACH PAGE (check reading order / column mixing) ===")
    seen = set()
    for l in lines:
        if l["page"] in seen:
            continue
        idx = [x for x in lines if x["page"] == l["page"]][:3]
        seen.add(l["page"])
        rep.append(f"  p{l['page']}:")
        for x in idx:
            rep.append(f"      {x['text'][:100]}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    text = "\n".join(rep)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n\nSaved to {OUT}  <-- paste this back")


if __name__ == "__main__":
    main()
