"""
STEP 2 — Parsing + Cleaning + Section detection + Section-aware chunking.

Design principle: NO hardcoded section names.
Headings are discovered from the document's own typography (font size clusters
relative to body text). That is why this works when a list of guessed strings
does not — it does not need to know what the ACG headings say, and it will work
on a second guideline too.

Run:   python 2_parse_chunk.py
Out:   data/processed/acg_chunks.json      <- feed this to embeddings
       data/processed/acg_sections.json    <- READ THIS to verify detection
"""

import glob
import json
import os
import re
import sys
from collections import Counter

import pymupdf

# Windows console is cp1252 and cannot print the "fi" ligature in this PDF.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ============================================================
# CONFIG
# ============================================================


def find_pdf():
    """Auto-locate the guideline PDF so a wrong path can't fail the run."""
    for pat in ("data/raw/*.pdf", "data/*.pdf", "*.pdf", "**/*.pdf"):
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    raise SystemExit("No PDF found. Put the ACG PDF in data/raw/ and re-run.")


PDF_PATH = find_pdf()
OUT_CHUNKS = "data/processed/acg_chunks.json"
OUT_SECTIONS = "data/processed/acg_sections.json"

CHUNK_TOKENS = 400      # embedding models truncate at 512 — never exceed it
OVERLAP_TOKENS = 60     # 15%
MIN_CHUNK_TOKENS = 40

DOCUMENT_ID = "ACG_2024"
SOURCE = "ACG Clinical Guideline 2024: Treatment of Helicobacter pylori Infection"
TOPIC = "Helicobacter pylori"

# ---- MEASURED from 1_inspect.py on this PDF ----
# body = 9.5, every real section heading = 10.0 ALL-CAPS  ->  ratio 1.053.
# The old 1.30 threshold never fired; only the ALL-CAPS rule did, and that
# rule alone also swallowed inline fragments (FDA, AST, RCT, PPI-, PCABs)).
HEADING_SIZE_RATIO = 1.04    # 10.0 / 9.5
TITLE_RATIO = 1.80           # 20.9 / 9.5 -> the article title, dropped
# Two different floors, because the two branches carry different risk:
#   - at heading SIZE (>= 10.0) almost nothing is a false positive, so 6 is
#     safe and it rescues METHODS (7 letters).
#   - BELOW heading size the only signal left is length, so demand 8. That
#     rescues ABSTRACT (8 letters, set at 9.0) and CONFLICTS OF INTEREST (8.5)
#     while still killing FDA(3) AST(3) RCT(3) CI(2) PPI(3) PCABs(5).
MIN_LETTERS_AT_SIZE = 6
MIN_LETTERS_BELOW_SIZE = 8
MAX_HEADING_WORDS = 16

try:
    from transformers import AutoTokenizer
    _T = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    def ntok(s): return len(_T.encode(s, add_special_tokens=False))
    TOKMODE = "exact (bge-small-en-v1.5)"
except Exception:
    def ntok(s): return int(len(s.split()) * 1.45)
    TOKMODE = "ESTIMATE (pip install transformers for exact)"


# ============================================================
# 1. EXTRACT — style-aware, column-aware, gap-aware
# ============================================================

def is_bold_font(span):
    """
    This PDF subsets its fonts, so the names are AdvOTa018106b.B / AdvPSOP-B /
    AdvOT08549422.BI — the word "bold" never appears. Checking for "bold" in the
    font name (the obvious approach) found zero bold text and every subsection
    heading was lost. Match the subset suffix instead.
    """
    if span["flags"] & 16:
        return True
    f = span["font"].lower()
    if "bold" in f or "black" in f or "semib" in f:
        return True
    return bool(re.search(r"[.\-](b|bi)$", f))


def order_blocks(blocks, page_width):
    """
    True two-column reading order.

    PyMuPDF's sort=True sorts by (y, x), which interleaves the two columns of a
    journal page. That is what put the REFERENCES list under the FUTURE RESEARCH
    PRIORITIES heading. Here: full-width blocks (tables, figures, headers) act
    as band separators; inside each band the left column is emitted top-to-
    bottom, then the right column.
    """
    mid = page_width / 2.0
    text_blocks = [b for b in blocks if b.get("type") == 0]
    text_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

    ordered, left, right = [], [], []

    def flush():
        ordered.extend(sorted(left, key=lambda b: b["bbox"][1]))
        ordered.extend(sorted(right, key=lambda b: b["bbox"][1]))
        left.clear()
        right.clear()

    for b in text_blocks:
        x0, x1 = b["bbox"][0], b["bbox"][2]
        if x0 < mid - 10 and x1 > mid + 10:        # spans both columns
            flush()
            ordered.append(b)
        elif x1 <= mid + 10:
            left.append(b)
        else:
            right.append(b)
    flush()
    return ordered


# ------------------------------------------------------------
# Table detection with a hard quality gate.
#
# find_tables() on this typeset journal PDF happily reports the page running
# header as a table:
#       | 1730 CLINICAL GUIDELINES |  |  |
# On page 1 that false table covered the ABSTRACT heading, so the heading was
# filtered out as "inside a table" and the whole abstract silently disappeared
# into dropped front matter. A bad table detection must never be allowed to
# delete prose. Everything below exists to enforce that.
# ------------------------------------------------------------

_TABLE_CACHE = {}

RUNNING_HEADER_RE = re.compile(
    r"(clinical guidelines|chey et al|american journal|volume \d+|"
    r"downloaded from|copyright)", re.I)


def collapse_empty_columns(rows):
    """
    find_tables() invents phantom columns on this layout: Table 1 comes back as
    6 columns of which 4 are entirely empty. Those phantoms made every real
    table fail the "rows must have 2 filled cells" test, so Table 5 and Table 6
    — the regimen tables — were rejected while the page header passed.
    Strip any column that is empty in more than 90% of rows, then judge.
    """
    if not rows:
        return rows
    width = max(len(r) for r in rows)
    padded = [list(r) + [""] * (width - len(r)) for r in rows]
    keep = [i for i in range(width)
            if sum(1 for r in padded if r[i].strip()) > 0.10 * len(padded)]
    if len(keep) < 1:
        return []
    return [[r[i] for i in keep] for r in padded]


def drop_caption_row(rows, caption):
    """find_tables() often repeats the caption as row 0. Remove the duplicate."""
    if rows and caption:
        first = " ".join(c for c in rows[0] if c).strip().lower()
        if first and first in caption.lower():
            return rows[1:]
    return rows


def _validate(rows, rect, page):
    """Return (ok, reason). Conservative on purpose — a missed table costs far
    less than a false one, because a false one deletes real text."""
    if len(rows) < 3:
        return False, "fewer than 3 rows"
    n_cols = max(len(r) for r in rows)
    if n_cols < 2:
        return False, "single column"

    # judged AFTER phantom columns are collapsed away
    # 0.25, measured: Table 6 passes at 35 rows, Table 5 sits at 10/32 = 0.31.
    # These regimen tables carry long footnote rows that legitimately occupy a
    # single cell, which drags the ratio down. The page-header false positive
    # is not admitted by loosening this — it is caught by RUNNING_HEADER_RE
    # immediately below.
    filled = [r for r in rows if sum(1 for c in r if c.strip()) >= 2]
    if len(filled) < 0.25 * len(rows):
        return False, f"only {len(filled)}/{len(rows)} rows have >=2 filled cells"
    if len(filled) < 2:
        return False, "fewer than 2 substantive rows"

    first = " ".join(rows[0])
    if RUNNING_HEADER_RE.search(first):
        return False, f"first row is page furniture: {first[:40]!r}"

    page_area = page.rect.get_area()
    if rect.get_area() > 0.90 * page_area:
        return False, "covers the whole page"
    if rect.get_area() < 0.02 * page_area:
        return False, "too small to be a table"
    return True, ""


def valid_tables(page):
    """Validated tables on this page: [{rect, rows, caption}]. Cached."""
    key = page.number
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]

    result = []
    try:
        found = list(page.find_tables().tables)
    except Exception:
        found = []

    for tbl in found:
        rect = pymupdf.Rect(tbl.bbox)
        caption = find_caption(page, rect)
        rows = collapse_empty_columns(table_to_rows(tbl))
        rows = drop_caption_row(rows, caption)
        ok, why = _validate(rows, rect, page)
        if not ok:
            REJECTED.append((page.number + 1, why, " ".join(rows[0])[:60]
                             if rows else ""))
            continue
        result.append({"rect": rect, "rows": rows, "caption": caption})

    _TABLE_CACHE[key] = result
    return result


REJECTED = []


def inside_any(rect, rects):
    """True if most of `rect` sits inside one of `rects`."""
    for r in rects:
        inter = rect & r
        if inter.is_valid and inter.get_area() > 0.6 * max(rect.get_area(), 1e-6):
            return True
    return False


def is_probable_heading_text(text):
    """Never let table exclusion swallow a section heading."""
    t = text.strip()
    letters = [c for c in t if c.isalpha()]
    if len(letters) < 6 or len(t.split()) > 12:
        return False
    return sum(c.isupper() for c in letters) / len(letters) > 0.75


def extract_lines(doc):
    """
    Prose lines only. Anything inside a detected table is removed here and
    handled by extract_tables() instead — otherwise the same dosing numbers
    appear twice: once scrambled into prose, once as a proper table.
    """
    out = []
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        t_rects = [t["rect"] for t in valid_tables(page)]
        blocks = page.get_text("dict")["blocks"]
        for block in order_blocks(blocks, page.rect.width):
            for ln in block["lines"]:
                if t_rects and inside_any(pymupdf.Rect(ln["bbox"]), t_rects):
                    raw = "".join(s["text"] for s in ln["spans"])
                    if not is_probable_heading_text(raw):
                        continue
                spans = [s for s in ln["spans"] if s["text"].strip()]
                if not spans:
                    continue

                # Join spans, inserting a space when they are visually apart.
                # Superscript affiliation markers split spans and glue words
                # together ("6Division ofGastroenterology") without this.
                parts, prev_x1, prev_sz = [], None, spans[0]["size"]
                for s in spans:
                    if prev_x1 is not None and s["bbox"][0] - prev_x1 > 0.18 * prev_sz:
                        if parts and not parts[-1].endswith(" "):
                            parts.append(" ")
                    parts.append(s["text"])
                    prev_x1, prev_sz = s["bbox"][2], s["size"]
                text = "".join(parts).strip()
                if not text:
                    continue

                sizes, bold, total = Counter(), 0, 0
                for s in spans:
                    n = len(s["text"])
                    sizes[round(s["size"], 1)] += n
                    if is_bold_font(s):
                        bold += n
                    total += n
                letters = [c for c in text if c.isalpha()]
                out.append({
                    "page": pno + 1,
                    "text": text,
                    "size": sizes.most_common(1)[0][0],
                    "bold": bold / max(total, 1) > 0.90,
                    "n_letters": len(letters),
                    "x0": round(ln["bbox"][0], 1),
                    "caps": bool(letters) and
                            sum(c.isupper() for c in letters) / len(letters) > 0.75,
                })
    return out


def body_size(lines):
    c = Counter()
    for l in lines:
        c[l["size"]] += len(l["text"])
    return c.most_common(1)[0][0]


# ============================================================
# 2. CLEAN
# ============================================================

LIG = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
       "’": "'", "‘": "'", "“": '"', "”": '"',
       "–": "-", " ": " "}

# Wolters Kluwer download stamp — appears inline mid-sentence on every page,
# not as a standalone line, so the whole-line NOISE rule never catches it.
WATERMARK_RE = re.compile(
    r"[A-Za-z0-9+/]{30,}={0,2}\s+on\s+\d{1,2}/\d{1,2}/\d{4}")

# Full lines that are journal furniture / front matter -> deleted.
NOISE = [
    r"^\d{1,4}$",
    r"american\s+journal\s+of\s+gastroenterolog",
    r"^volume\s+\d+\s*\|",
    r"amjgastro\.com",
    r"^clinical\s+guidelines?$",
    r"^acg\s+clinical\s+guideline$",
    r"downloaded\s+from",
    r"unauthorized\s+reproduction",
    r"^copyright",
    r"^©\s*\d{4}",
    r"^correspondence\s*:",
    r"^e-?mail\s*:",
    r"^received\s+\w+\s+\d{1,2},",
    r"^(https?://|doi:|www\.)",
    r"^[A-Za-z0-9+/]{35,}={0,2}$",          # Wolters Kluwer tracking token
    r"^=?\s*on\s+\d{1,2}/\d{1,2}/\d{4}$",   # leftover download stamp
    r"^\d*\s*(division|department)\s+of\b.*\b(usa|university|hospital|center|medicine)",
    r"^\d*\s*(university|college|school)\s+of\b.*\busa\b",
    r"\bUSA[;.]\s*$",                        # affiliation tail lines
    r"^supplementary\s+material$",
    r"^see\s+figure\s+\d",
    r"^\(?see\s+table\s+\d",
    # running headers found by 1_inspect.py on every page
    r"^chey\s+et\s+al\.?$",
    r"^treatment\s+of\s+helicobacter\s+pylori\s+infection$",
]
NOISE_RE = [re.compile(p, re.I) for p in NOISE]

# Hyphens that are part of the real term and must survive line-wrap joining.
KEEP_HYPHEN = {
    "treatment", "first", "second", "third", "post", "pre", "non", "anti", "co",
    "clarithromycin", "levofloxacin", "penicillin", "proton", "potassium",
    "high", "low", "double", "triple", "quadruple", "long", "short", "well",
    "self", "acid", "cost", "meta", "follow", "up", "salvage",
}


def is_noise(t):
    t = t.strip()
    return not t or any(r.search(t) for r in NOISE_RE)


def clean(t):
    for a, b in LIG.items():
        t = t.replace(a, b)
    t = WATERMARK_RE.sub("", t)
    t = re.sub(r"([,;:])(?=[A-Za-z])", r"\1 ", t)          # "Intervention,Comparison"
    t = re.sub(r"(?<=[a-z]{2})(?=[A-Z][a-z]{2})", " ", t)  # "ofGastroenterology"
    t = re.sub(r"(?<=[a-z])\d{1,2}(?=[\s,.;)])", "", t)    # stray superscript refs
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def join_wrapped(prev, nxt):
    """Join a line ending in '-' with the next, preserving real compounds."""
    stem = prev[:-1]
    last = re.split(r"[^A-Za-z]", stem)[-1].lower()
    return stem + ("-" if last in KEEP_HYPHEN else "") + nxt


# ============================================================
# 3. SECTION DETECTION — from typography, not from a name list
# ============================================================

def heading_level(l, body):
    """
    Measured rules for this document (see 1_inspect.py output):

      body text ............ 9.5
      section headings ..... 10.0 + ALL CAPS      -> level 1
      article title ........ 20.9                 -> title, dropped
      subsection headings .. body size + bold     -> level 2

    The false positives that wrecked the previous run were all ALL-CAPS
    fragments at or below body size with fewer than 8 letters, or ending in
    punctuation: FDA, AST, RCT, CI, H., PPI-, PCABs), EMBASE,, MEDLINE,.
    Both filters below are aimed squarely at them.
    """
    t = l["text"].strip()
    if len(t.split()) > MAX_HEADING_WORDS or len(t) < 3:
        return 0
    if not any(c.isalpha() for c in t):
        return 0
    # Real headings never end in sentence punctuation or a dangling bracket.
    if t.rstrip().endswith((".", ",", ";", ":", ")", "]", "-", "/")):
        return 0
    if re.match(r"^(table|figure|fig\.)\s*\d", t, re.I):
        return 0                                  # captions are not sections
    if re.search(r"\d{2,}", t):                   # "(84.5%,653/773)" etc.
        return 0

    r = l["size"] / body

    if r >= TITLE_RATIO:
        return 9                                  # article title -> front matter

    # ---- level 1: ALL CAPS ----
    if l["caps"]:
        if r >= HEADING_SIZE_RATIO:
            # at heading size: METHODS (7), EPIDEMIOLOGY (12), REFERENCES (10)
            return 1 if l["n_letters"] >= MIN_LETTERS_AT_SIZE else 0
        # below heading size, length is the only signal we have left:
        # ABSTRACT (8 @ 9.0), CONFLICTS OF INTEREST (19 @ 8.5)
        return 1 if l["n_letters"] >= MIN_LETTERS_BELOW_SIZE else 0

    # ---- level 2: fully bold, short, sentence-case ----
    # ACG structures each recommendation as bold run-in headings
    # ("Recommendation", "Key concept", "Summary of the evidence").
    if l["bold"] and len(t.split()) <= 10 and l["n_letters"] >= 4:
        return 2

    return 0


def segment(lines, body):
    """
    Walk EVERY line. No per-page break — that bug is what swallowed most of the
    sections in the earlier version. Consecutive heading lines are merged so a
    heading wrapped across two lines is recovered as one.
    """
    sections = []
    cur = {"section": "FRONT MATTER", "subsection": "", "level": 1,
           "page_start": 1, "page_end": 1, "lines": []}
    i = 0
    while i < len(lines):
        l = lines[i]
        if is_noise(l["text"]):
            i += 1
            continue

        lvl = heading_level(l, body)

        # ---- orphaned heading tails ----
        # A bold subsection heading that wraps across a COLUMN break leaves its
        # second line stranded at the top of the next column, with a whole
        # column of body text in between. It then looked like a brand-new
        # subsection: "malignant conditions", "experienced patients",
        # "approval". The giveaway is that a real heading never starts with a
        # lowercase word. Re-attach it to the heading it belongs to.
        if lvl == 2 and l["text"].strip()[:1].islower():
            if cur["subsection"]:
                cur["subsection"] = clean(cur["subsection"] + " " + l["text"].strip())
            else:
                cur["lines"].append(l)      # no heading to attach to -> body text
            cur["page_end"] = l["page"]
            i += 1
            continue

        if lvl:
            # merge consecutive same-level heading lines (wrapped headings)
            group, j = [l["text"]], i + 1
            while j < len(lines) and j - i < 3:
                nxt = lines[j]
                if is_noise(nxt["text"]) or heading_level(nxt, body) != lvl:
                    break
                if abs(nxt["size"] - l["size"]) > 0.6:
                    break
                group.append(nxt["text"])
                j += 1
            title = clean(" ".join(group))

            if cur["lines"]:
                sections.append(cur)
            if lvl == 9:                     # article title + author block
                parent, sub = "FRONT MATTER", ""
            elif lvl == 2:
                parent, sub = cur["section"], title
            else:
                parent, sub = title, ""
            cur = {"section": parent, "subsection": sub, "level": lvl,
                   "page_start": l["page"], "page_end": l["page"], "lines": []}
            i = j
            continue

        cur["lines"].append(l)
        cur["page_end"] = l["page"]
        i += 1

    if cur["lines"]:
        sections.append(cur)

    # stitch lines into prose
    for s in sections:
        buf = ""
        for l in s["lines"]:
            t = l["text"].rstrip()
            if buf.endswith("-"):
                buf = join_wrapped(buf, t.lstrip())
            else:
                buf = (buf + " " + t) if buf else t
        s["text"] = clean(buf)
        del s["lines"]

    # merge heading-only / stub sections into the previous one
    merged = []
    for s in sections:
        if merged and len(s["text"]) < 150 and s["level"] == 2:
            p = merged[-1]
            p["text"] = (p["text"] + " " + s["subsection"] + " " + s["text"]).strip()
            p["page_end"] = max(p["page_end"], s["page_end"])
        else:
            merged.append(s)
    return [s for s in merged if s["text"].strip()]


# ============================================================
# 3b. SPLIT EMBEDDED SUMMARY TABLES (Tables 2, 3, 4)
# ============================================================
# find_tables() does not detect Tables 2-4 on this journal layout. Two-column
# reading order interleaves their captions and rows into EPIDEMIOLOGY prose,
# which is why ACG_0010 carried recommendation-table text under "Key concept".

TABLE2_RE = re.compile(r"Table\s+2\.?\s*Guideline recommendations", re.I)
TABLE3_RE = re.compile(r"Table\s+3\.?\s*Summary of key concepts", re.I)
TABLE4_RE = re.compile(
    r"Table\s+4\.?\s*Indications for H\. pylori testing and treatment", re.I)
NAIVE_HDR = re.compile(
    r"Recommendations for treatment-naive patients with Helicobacter pylori infection",
    re.I)
EXPER_HDR = re.compile(
    r"Recommendations for treatment-experienced patients with persistent H\. pylori infection",
    re.I)
# Column interleave leaves a dangling "The global" before Table 2 and epidemiology
# prose after the table abbreviation footer.
ORPHAN_GLOBAL = re.compile(r"\s+The global\s*$")
TABLE_FOOTER = re.compile(
    r"BQT, bismuth quadruple therapy; PCAB, potassium-competitive acid blocker; "
    r"PICO, Population, Intervention, Comparison, and Outcome; PPI, proton pump inhibitor\.\s*",
    re.I)
EPIDEMIOLOGY_RESUME = re.compile(
    r"^(prevalence has declined|Chronic gastric infection|In North America|"
    r"The highest seroprevalence|Intrafamilial person-to-person)",
    re.I)

SECTION_NAIVE = "ERADICATING HELICOBACTER PYLORI INFECTION IN TREATMENT-NAIVE PATIENTS"
SECTION_EXPER = "ERADICATING HELICOBACTER PYLORI INFECTION IN TREATMENT-EXPERIENCED PATIENTS"
SECTION_INDICATIONS = "INDICATIONS FOR HELICOBACTER PYLORI TESTING AND TREATMENT"


def _section_piece(text, section, subsection, level, page_start, page_end,
                   content_type="prose"):
    return {
        "section": section,
        "subsection": subsection,
        "level": level,
        "page_start": page_start,
        "page_end": page_end,
        "text": clean(text),
        "content_type": content_type,
    }


def _split_table2_block(text, page_start, page_end):
    """Table 2 summary -> treatment-naive and treatment-experienced sections."""
    parts = []
    m_exp = EXPER_HDR.search(text)
    if m_exp:
        naive_txt = text[:m_exp.start()].strip()
        exper_txt = text[m_exp.start():].strip()
    else:
        naive_txt, exper_txt = text, ""

    if naive_txt:
        parts.append(_section_piece(
            naive_txt, SECTION_NAIVE, "Table 2. Guideline recommendations",
            2, page_start, page_end, "table_summary"))
    if exper_txt:
        parts.append(_section_piece(
            exper_txt, SECTION_EXPER, "Table 2. Guideline recommendations",
            2, page_start, page_end, "table_summary"))
    return parts


def _split_at_captions(text):
    """Return [(caption_re, start, end), ...] sorted by position."""
    hits = []
    for rx in (TABLE2_RE, TABLE3_RE, TABLE4_RE):
        for m in rx.finditer(text):
            hits.append((rx, m.start(), m.end()))
    hits.sort(key=lambda x: x[1])
    return hits


def _assign_table_block(caption_re, block, page_start, page_end):
    if caption_re is TABLE2_RE:
        return _split_table2_block(block, page_start, page_end)
    if caption_re is TABLE3_RE:
        return [_section_piece(
            block, "SUMMARY OF KEY CONCEPTS",
            "Table 3. Summary of key concepts",
            2, page_start, page_end, "table_summary")]
    if caption_re is TABLE4_RE:
        return [_section_piece(
            block, SECTION_INDICATIONS,
            "Table 4. Indications for H. pylori testing and treatment",
            2, page_start, page_end, "table_summary")]
    return []


def _detach_interleaved_epidemiology(text, page_start, page_end):
    """
    Table 2's abbreviation footer is immediately followed by epidemiology prose
    from the other column ('prevalence has declined...'). Pull it back out.
    """
    m = TABLE_FOOTER.search(text)
    if not m:
        return text, None
    tail = text[m.end():].strip()
    if not tail or not EPIDEMIOLOGY_RESUME.match(tail):
        return text, None
    table_part = text[:m.end()].strip()
    epi = _section_piece(
        tail, "EPIDEMIOLOGY", "Key concept", 2, page_start, page_end)
    return table_part, epi


def split_embedded_tables(sections):
    """Re-file Tables 2-4 that bled into EPIDEMIOLOGY via column interleaving."""
    out = []
    for sec in sections:
        text = sec["text"]
        hits = _split_at_captions(text)
        if not hits:
            out.append(sec)
            continue

        pos = 0
        for i, (rx, start, end) in enumerate(hits):
            prefix = text[pos:start].strip()
            if prefix:
                prefix = ORPHAN_GLOBAL.sub("", prefix).strip()
                if prefix:
                    piece = dict(sec)
                    piece["text"] = clean(prefix)
                    out.append(piece)

            nxt = hits[i + 1][1] if i + 1 < len(hits) else len(text)
            block = text[start:nxt].strip()
            block, epi_tail = _detach_interleaved_epidemiology(
                block, sec["page_start"], sec["page_end"])
            out.extend(_assign_table_block(
                rx, block, sec["page_start"], sec["page_end"]))
            if epi_tail:
                out.append(epi_tail)
            pos = nxt

        suffix = text[pos:].strip()
        if suffix:
            # Trailing epidemiology after the last embedded table on this page.
            if TABLE3_RE.search(suffix) or TABLE4_RE.search(suffix):
                out.extend(split_embedded_tables([dict(sec, text=suffix)]))
            else:
                piece = dict(sec)
                piece["text"] = clean(suffix)
                out.append(piece)

    return [s for s in out if s.get("text", "").strip()]


# ============================================================
# 4. SECTION-AWARE CHUNKING
# ============================================================

SENT = re.compile(
    r"(?<!\be\.g)(?<!\bi\.e)(?<!\bvs)(?<!\bFig)(?<!\bNo)(?<!\bDr)"
    r"(?<!\b[A-Z])(?<=[.!?])\s+(?=[A-Z(\[])"
)


def sentences(text):
    out = []
    for s in SENT.split(text):
        s = s.strip()
        if not s:
            continue
        if ntok(s) <= CHUNK_TOKENS:
            out.append(s)
        else:                                    # long table/list block, no periods
            # Check BEFORE appending, otherwise the emitted piece lands just
            # over the budget instead of just under it.
            cur = []
            for w in s.split():
                if cur and ntok(" ".join(cur + [w])) > CHUNK_TOKENS:
                    out.append(" ".join(cur))
                    cur = [w]
                else:
                    cur.append(w)
            if cur:
                out.append(" ".join(cur))
    return out


# Sections that answer no clinical question. Dropped before chunking.
DROP_SECTIONS = {"FRONT MATTER", "REFERENCES", "CONFLICTS OF INTEREST",
                 "ACKNOWLEDGMENTS", "ACKNOWLEDGEMENTS"}


def looks_like_reference_list(text):
    """
    Content-based detector, because column order can put the reference list
    under the wrong heading (that is exactly what happened: 28,519 characters
    of references were labelled FUTURE RESEARCH PRIORITIES).

    A reference list has a very high density of "et al" and year;volume:page
    citations compared with any prose section.
    """
    if len(text) < 2000:
        return False
    per_1k = 1000.0 / len(text)
    et_al = text.lower().count("et al") * per_1k
    cites = len(re.findall(r"\d{4};\s*\d+", text)) * per_1k
    return et_al >= 1.0 or cites >= 1.0


# ============================================================
# 4b. TABLES — extracted as tables, never flattened into prose
# ============================================================
# The regimen and dosing tables (Table 2, Table 5, Table 6) are the most
# clinically important content in this guideline. Flattened into prose, a row
# reads as fluent text while the drug, its dose and its duration have been
# separated — which is worse than not indexing them at all.

CAPTION_RE = re.compile(r"^\s*(Table|Figure)\s*(\d+)\s*[.:]?\s*(.*)$", re.I)


def find_caption(page, table_rect):
    """Nearest 'Table N. ...' line sitting just above the table."""
    best, best_gap = "", 1e9
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for ln in block["lines"]:
            text = "".join(s["text"] for s in ln["spans"]).strip()
            m = CAPTION_RE.match(text)
            if not m:
                continue
            # find_tables() usually includes the caption inside the table
            # bbox, so allow it to sit slightly BELOW the top edge too.
            gap = table_rect.y0 - ln["bbox"][3]
            if -60 <= gap < 120 and abs(gap) < abs(best_gap):
                best, best_gap = text, gap
    return clean(best)


def cell(v):
    return re.sub(r"\s+", " ", clean(str(v or ""))).strip()


def table_to_rows(tbl):
    rows = []
    for r in tbl.extract():
        cells = [cell(c) for c in r]
        if any(cells):
            rows.append(cells)
    return rows


def extract_tables(doc):
    """One chunk per table; large tables split by rows with the header repeated."""
    out, tid = [], 0
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        for t in valid_tables(page):
            rows = t["rows"]
            tid += 1
            caption = t["caption"] or f"Unlabelled table on page {pno + 1}"
            header = rows[0]
            head_md = "| " + " | ".join(header) + " |"
            sep_md = "|" + "|".join("---" for _ in header) + "|"
            prefix = f"{caption}\n{head_md}\n{sep_md}"

            body, buf = rows[1:], []
            parts = []

            def flush_rows():
                if buf:
                    parts.append("\n".join(buf))
                    buf.clear()

            for r in body:
                line = "| " + " | ".join(r) + " |"
                trial = prefix + "\n" + "\n".join(buf + [line])
                # Never split a row. If adding it overflows, close the chunk.
                if buf and ntok(trial) > CHUNK_TOKENS:
                    flush_rows()
                buf.append(line)
            flush_rows()

            for i, part in enumerate(parts):
                text = prefix + "\n" + part
                out.append({
                    "text": text,
                    "caption": caption,
                    "page": pno + 1,
                    "table_index": tid,
                    "part": i,
                    "parts": len(parts),
                    "n_rows": len(part.splitlines()),
                })
    return out


def section_for_page(sections, page):
    """
    A table belongs to the section it physically sits in, not to a flat
    "TABLES" bucket. Table 5 (p8) is part of TREATMENT-NAIVE; Table 6 (p14) is
    part of TREATMENT-EXPERIENCED. Filing them under "TABLES" made the
    retriever look wrong when it was actually right — it returned the regimen
    table for a regimen question and the eval scored it as a miss.
    """
    best = None
    for s in sections:
        if s["section"].upper() in DROP_SECTIONS:
            continue
        if s["page_start"] <= page and (best is None
                                        or s["page_start"] >= best["page_start"]):
            best = s
    return best["section"] if best else "TABLES"


def table_chunks(doc, start_id, sections):
    chunks, cid = [], start_id
    for t in extract_tables(doc):
        chunks.append({
            "chunk_id": f"ACG_T{cid:04d}",
            "document_id": DOCUMENT_ID,
            "text": t["text"],
            "section": section_for_page(sections, t["page"]),
            "subsection": t["caption"][:120],
            "content_type": "table",
            "table_index": t["table_index"],
            "table_part": f"{t['part'] + 1}/{t['parts']}",
            "n_rows": t["n_rows"],
            "page_start": t["page"],
            "page_end": t["page"],
            "n_tokens": ntok(t["text"]),
            "source": SOURCE,
            "topic": TOPIC,
            "citation": f"{SOURCE} — {t['caption'][:80]} (p. {t['page']})",
        })
        cid += 1
    return chunks


def chunk(sections):
    """A chunk NEVER crosses a section boundary. Overlap stays inside a section."""
    chunks, cid = [], 1
    for sec in sections:
        if sec["section"].upper() in DROP_SECTIONS:
            continue
        if looks_like_reference_list(sec["text"]):
            print(f"[drop] '{sec['section'][:50]}' p{sec['page_start']}-"
                  f"{sec['page_end']} ({len(sec['text'])} ch) "
                  f"detected as a reference list")
            continue
        sents = sentences(sec["text"])
        buf, tok = [], 0

        def emit():
            nonlocal cid, buf
            if not buf:
                return
            text = " ".join(buf)
            if ntok(text) < MIN_CHUNK_TOKENS:
                return
            cite = f"{SOURCE} — {sec['section']}"
            if sec["subsection"]:
                cite += f" / {sec['subsection']}"
            cite += f" (p. {sec['page_start']})"
            chunks.append({
                "chunk_id": f"ACG_{cid:04d}",
                "document_id": DOCUMENT_ID,
                "text": text,
                "section": sec["section"],
                "subsection": sec["subsection"],
                "page_start": sec["page_start"],
                "page_end": sec["page_end"],
                "n_tokens": ntok(text),
                "source": SOURCE,
                "topic": TOPIC,
                "citation": cite,
                "content_type": sec.get("content_type", "prose"),
            })
            cid += 1

        for s in sents:
            t = ntok(s)
            if buf and tok + t > CHUNK_TOKENS:
                emit()
                tail, tt = [], 0
                for x in reversed(buf):
                    xt = ntok(x)
                    if tt + xt > OVERLAP_TOKENS and tail:
                        break
                    tail.insert(0, x)
                    tt += xt
                # The carried overlap plus the incoming sentence can itself
                # exceed the budget (long recommendation blocks). Shrink the
                # overlap from the front until it fits — the new sentence
                # matters more than the repeated context.
                while tail and tt + t > CHUNK_TOKENS:
                    tt -= ntok(tail[0])
                    tail.pop(0)
                buf, tok = tail, tt
            buf.append(s)
            tok += t
        emit()
    return chunks


# ============================================================
# 5. RUN + VALIDATE
# ============================================================

if __name__ == "__main__":
    print("PDF:", PDF_PATH)
    doc = pymupdf.open(PDF_PATH)
    lines = extract_lines(doc)
    body = body_size(lines)
    sections = segment(lines, body)
    sections = split_embedded_tables(sections)
    chunks = chunk(sections)
    for c in chunks:
        if "content_type" not in c:
            c["content_type"] = "prose"

    tbls = table_chunks(doc, len(chunks) + 1, sections)
    chunks = chunks + tbls

    os.makedirs(os.path.dirname(OUT_CHUNKS), exist_ok=True)
    json.dump(chunks, open(OUT_CHUNKS, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump([{k: v for k, v in s.items() if k != "text"} |
               {"chars": len(s["text"]), "preview": s["text"][:160]}
               for s in sections],
              open(OUT_SECTIONS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    assert chunks, "No chunks. Check the PDF."

    # Last line of defence. An over-budget chunk is silently truncated by the
    # embedding model, so trim it here where it is visible instead.
    # Tables are NEVER trimmed — cutting a dosing row is the exact harm this
    # whole table path exists to prevent. They are reported loudly instead.
    fat_tables = [c for c in chunks
                  if c["content_type"] == "table" and c["n_tokens"] > CHUNK_TOKENS]
    for c in fat_tables:
        print(f"[warn] table chunk {c['chunk_id']} is {c['n_tokens']} tokens "
              f"(> {CHUNK_TOKENS}) — {c['subsection'][:60]}. "
              f"It will be truncated by the embedder; review it by hand.")

    over = [c for c in chunks
            if c["content_type"] == "prose" and c["n_tokens"] > CHUNK_TOKENS]
    for c in over:
        words = c["text"].split()
        while words and ntok(" ".join(words)) > CHUNK_TOKENS:
            words.pop()
        c["text"] = " ".join(words)
        c["n_tokens"] = ntok(c["text"])
        c["trimmed"] = True
    if over:
        print(f"[warn] trimmed {len(over)} over-budget chunk(s): "
              f"{[c['chunk_id'] for c in over]}")
        json.dump(chunks, open(OUT_CHUNKS, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    assert all(c["n_tokens"] <= CHUNK_TOKENS
               for c in chunks if c["content_type"] == "prose"), "trim failed"

    toks = sorted(c["n_tokens"] for c in chunks)
    print("=" * 60)
    print("ACG — PARSE / CLEAN / SECTION / CHUNK")
    print("=" * 60)
    print("Pages           :", doc.page_count)
    print("Body font size  :", body)
    print("Token counting  :", TOKMODE)
    n_prose = sum(1 for c in chunks if c["content_type"] == "prose")
    n_table = sum(1 for c in chunks if c["content_type"] == "table")
    print("Sections        :", len(sections))
    print("Chunks          :", len(chunks), f"({n_prose} prose + {n_table} table)")
    print("Chunk budget    :", CHUNK_TOKENS, "tokens | overlap:", OVERLAP_TOKENS)
    print(f"Chunk tokens    : min={toks[0]} median={toks[len(toks)//2]} max={toks[-1]}")

    print("\n--- SECTIONS DETECTED ---")
    kept = 0
    for s in sections:
        dropped = (s["section"].upper() in DROP_SECTIONS
                   or looks_like_reference_list(s["text"]))
        mark = "DROP" if dropped else "    "
        if not dropped:
            kept += 1
        lbl = s["section"] + (f"  /  {s['subsection']}" if s["subsection"] else "")
        print(f"  {mark} L{s['level']}  p{s['page_start']:>2}-{s['page_end']:<2}  "
              f"{len(s['text']):>6} ch   {lbl[:95]}")
    print(f"\n  kept {kept} sections, dropped {len(sections) - kept}")

    print("\n--- TOP-LEVEL SECTIONS ---")
    for name in dict.fromkeys(s["section"] for s in sections):
        if name.upper() in DROP_SECTIONS:
            continue
        n = sum(1 for c in chunks if c["section"] == name)
        print(f"  {n:>3} chunks   {name}")

    print("\n--- TABLES EXTRACTED ---")
    seen_t = {}
    for c in chunks:
        if c["content_type"] != "table":
            continue
        seen_t.setdefault(c["table_index"], []).append(c)
    if not seen_t:
        print("  none passed validation")
    for idx, parts in sorted(seen_t.items()):
        p = parts[0]
        rows = sum(x["n_rows"] for x in parts)
        print(f"  p{p['page_start']:>2}  {len(parts)} chunk(s)  {rows:>3} rows   "
              f"{p['subsection'][:72]}")
        print(f"        filed under: {p['section'][:70]}")

    print("\n--- ALL TABLE CHUNKS (read these, they carry the doses) ---")
    for c in chunks:
        if c["content_type"] == "table":
            print(f"\n### {c['chunk_id']}  part {c['table_part']}  p{c['page_start']}")
            print(c["text"][:1100])

    if REJECTED:
        print("\n--- TABLE CANDIDATES REJECTED (left in prose, nothing lost) ---")
        for pg, why, first in REJECTED:
            print(f"  p{pg:>2}  {why:<45} {first}")

    tbl_example = next((c for c in chunks if c["content_type"] == "table"), None)
    if tbl_example:
        print("\n--- FIRST TABLE CHUNK (verify the rows line up) ---")
        print(tbl_example["text"][:900])

    print("\nSaved:", OUT_CHUNKS)
    print("Saved:", OUT_SECTIONS, " <-- verify this before embedding")
    print("\n--- FIRST CHUNK ---")
    print(json.dumps(chunks[0], ensure_ascii=False, indent=2)[:1500])
