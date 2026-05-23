"""
Parse finland_kb into nodes.json.
Filters to tax-relevant files by keyword match on filename + content.
Usage:
  python parser.py                  # parse all, write data/nodes.json
  python parser.py --limit 500      # cap at N nodes for testing
  python parser.py --stats          # print stats only, don't write
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

KB_ROOT = Path("finland_kb")
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "nodes.json"

# Tax-relevant keyword filter — matches filename OR body text
TAX_KEYWORDS_RE = re.compile(
    r"vero|tulo|ALV|arvonlisä|TVL|AVL|EPL|VML|elinkeinoverolaki|perintö|lahja|"
    r"kiinteistö|pääoma|ennakon|pidätys|lähdever|yhteisöver|osinko|palkka|"
    r"vuokra|myyntivoitto|ulkomainen|rajoitetusti verovelvollinen|key.personnel|"
    r"avainhenkilö|withholding|eläke|TyEL|MEL|YEL|MYEL|vakuuttamisvelvoll|"
    r"työeläke|kansaneläke|sosiaaliturva",
    re.IGNORECASE,
)

# Chunking parameters: keep within multilingual-mpnet's ~512-token window
# 800 chars ≈ 200-270 Finnish tokens, leaves headroom after title prefix
CHUNK_CHARS = 800
CHUNK_OVERLAP = 150
CHUNK_MIN_CHARS = 120

# Section-boundary heuristics: numbered headings (2.7, 2.7.2), or ALL-CAPS / Title-Case lines
SECTION_BOUNDARY_RE = re.compile(
    r"(?=\n\s*\d+(?:\.\d+){0,3}\s+[A-ZÄÖÅ])"  # 2 X, 2.7 X, 2.7.2 X
    r"|(?=\n\s*\d+\s*§\s)"                      # statute § markers
    r"|(?=\n\s*[A-ZÄÖÅ][A-ZÄÖÅ ]{4,}\n)"        # ALL CAPS headings
)


def chunk_text(text: str) -> list[str]:
    """Split into ~CHUNK_CHARS-sized chunks, preferring section boundaries."""
    if len(text) <= CHUNK_CHARS:
        return [text]

    # First-pass split on section boundaries
    sections = SECTION_BOUNDARY_RE.split(text)
    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= CHUNK_CHARS:
            chunks.append(section)
            continue
        # Section still too long — sliding window
        start = 0
        while start < len(section):
            chunk = section[start : start + CHUNK_CHARS]
            if len(chunk.strip()) >= CHUNK_MIN_CHARS:
                chunks.append(chunk)
            start += CHUNK_CHARS - CHUNK_OVERLAP
    return chunks or [text[:CHUNK_CHARS]]

# Finnish § notation: "12 §", "§ 12", "12 §:n", "12 §:ään", "§:n 12"
REF_RE = re.compile(r"(\d+[a-zA-Z]?)\s*§|§\s*(\d+[a-zA-Z]?)", re.IGNORECASE)

# Extract effective date from text (e.g. "1.1.2024", "2024-01-01")
DATE_RE = re.compile(r"\b(20\d\d)[.\-/](0?\d|1[0-2])[.\-/](0?\d|[12]\d|3[01])\b")


def extract_date(text: str) -> str | None:
    m = DATE_RE.search(text)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return None


def extract_references(text: str, statute: str | None) -> list[str]:
    """Extract cross-references like §33, §102 from text."""
    refs = []
    for m in REF_RE.finditer(text):
        sec = m.group(1) or m.group(2)
        if sec and statute:
            refs.append(f"{statute}_§{sec}")
    return list(dict.fromkeys(refs))  # dedupe, preserve order


def classify_finlex_type(folder: str) -> str:
    folder = folder.lower()
    if "korkein hallinto" in folder or "kho" in folder:
        return "COURT_CASE"
    if "asetus" in folder:
        return "STATUTE"
    return "ARTICLE"


def classify_vero_type(folder: str) -> str:
    folder = folder.lower()
    if "ennakkoratkaisu" in folder or "kvl" in folder:
        return "COURT_CASE"
    return "GUIDANCE"


def infer_statute(title: str, filepath: str) -> str | None:
    """Try to extract statute abbreviation from title or path.
    Uses word stems to handle Finnish morphological declension
    (tuloverolaki/tuloverolain/tuloverolakia etc.)
    """
    checks = [
        (r"\bTVL\b", "TVL"),
        (r"\bAVL\b", "AVL"),
        (r"\bEPL\b", "EPL"),
        (r"\bVML\b", "VML"),
        (r"\bTyEL\b", "TyEL"),
        (r"\bMEL\b", "MEL"),
        (r"\bYEL\b", "YEL"),
        (r"\bMYEL\b", "MYEL"),
        (r"tuloverola",         "TVL"),   # stem covers all case forms
        (r"arvonlisäverola",   "AVL"),
        (r"ennakkoperintä",    "EPL"),
        (r"verotusmenettel",   "VML"),
        (r"työntekijän eläkelak",  "TyEL"),
        (r"työntekijäin eläkelak", "TyEL"),
        (r"merimieseläkelak",      "MEL"),
        (r"yrittäjän eläkelak",    "YEL"),
        (r"maatalousyrittäj.*eläkelak", "MYEL"),
        (r"perintö.*lahja",    "PerVL"),
        (r"kiinteistöver",     "KiVL"),
        (r"elinkeinoverolak",  "EVL"),
        (r"lähdeverola",       "LähdeVL"),
        (r"korkotulojen.*lähdever", "KoroVL"),
        (r"varojen siirtymis", "PerVL"),
    ]
    combined = (title + " " + filepath).lower()
    for pattern, abbrev in checks:
        if re.search(pattern, combined, re.IGNORECASE):
            return abbrev
    return None


def parse_html(path: Path) -> list[dict] | None:
    """Parse a single HTML file into one or more node dicts (chunked).
    Returns None if not tax-relevant.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  SKIP (read error): {path} — {e}", file=sys.stderr)
        return None

    # Fast keyword pre-filter on filename + raw text (avoid full BeautifulSoup parse)
    combined_for_filter = str(path) + raw[:2000]
    if not TAX_KEYWORDS_RE.search(combined_for_filter):
        return None

    soup = BeautifulSoup(raw, "html.parser")

    # Title: prefer <h1>, fallback to <title> or filename stem
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else (
        soup.find("title").get_text(strip=True) if soup.find("title") else path.stem
    )

    # Body text: extract from semantic tags; fall back to full body if sparse
    body_parts = []
    for tag in soup.find_all(["p", "h2", "h3", "h4", "li", "td", "th", "div"]):
        # Skip nested — only take leaf-ish text to avoid duplication
        if tag.find(["p", "h2", "h3", "h4", "li", "td", "div"]):
            continue
        t = tag.get_text(separator=" ", strip=True)
        if t:
            body_parts.append(t)
    text = " ".join(body_parts)
    # If still sparse, fall back to full body text
    if len(text) < 200 and soup.body:
        text = soup.body.get_text(separator=" ", strip=True)

    if not text.strip():
        return None

    # Determine source + type based on path
    rel = path.relative_to(KB_ROOT)
    parts = rel.parts
    source = parts[0]  # "finlex" or "vero"
    folder = parts[1] if len(parts) > 1 else ""

    if source == "finlex":
        node_type = classify_finlex_type(folder)
    else:
        node_type = classify_vero_type(folder)

    statute = infer_statute(title, str(path))

    # Stable node ID from path
    node_id = str(rel).replace("/", "_").replace(" ", "_").replace(".html", "")
    # Trim to reasonable length
    if len(node_id) > 120:
        node_id = node_id[:120]

    date = extract_date(text)

    # Chunk long documents — each chunk becomes its own retrievable node
    chunks = chunk_text(text)
    nodes: list[dict] = []
    for i, chunk in enumerate(chunks):
        chunk_refs = extract_references(chunk, statute)
        suffix = "" if len(chunks) == 1 else f"#chunk{i}"
        nodes.append({
            "id": (node_id + suffix)[:140],
            "parent_id": node_id,
            "chunk_index": i,
            "chunk_total": len(chunks),
            "source": source,
            "type": node_type,
            "statute": statute,
            "title": title,
            "text": chunk,
            "date": date,
            "superseded_by": None,
            "references": chunk_refs,
            "file_path": str(path),
        })
    return nodes


def parse_corpus(limit: int | None = None) -> list[dict]:
    nodes = []
    skipped = 0
    total = 0

    # Process vero first (always small), then finlex — so vero is never squeezed by --limit
    vero_html = sorted((KB_ROOT / "vero").rglob("*.html"))
    finlex_html = sorted((KB_ROOT / "finlex").rglob("*.html"))
    all_html = vero_html + finlex_html
    print(f"Found {len(all_html)} HTML files ({len(vero_html)} vero + {len(finlex_html)} finlex). Filtering...\n")

    for path in all_html:
        total += 1
        node_chunks = parse_html(path)
        if not node_chunks:
            skipped += 1
            continue
        nodes.extend(node_chunks)
        if len(nodes) % 500 == 0:
            print(f"  Parsed {len(nodes)} chunks from {total - skipped} files ({total} scanned, {skipped} skipped)...")
        if limit and len(nodes) >= limit:
            break

    print(f"\nDone. {len(nodes)} tax-relevant nodes from {total} files ({skipped} skipped).")
    return nodes


def print_stats(nodes: list[dict]):
    from collections import Counter
    types = Counter(n["type"] for n in nodes)
    sources = Counter(n["source"] for n in nodes)
    statutes = Counter(n["statute"] for n in nodes if n["statute"])
    print("\n--- Node stats ---")
    print("By type:", dict(types))
    print("By source:", dict(sources))
    print("Top statutes:", statutes.most_common(10))
    avg_len = sum(len(n["text"]) for n in nodes) / len(nodes) if nodes else 0
    print(f"Avg text length: {avg_len:.0f} chars")
    with_refs = sum(1 for n in nodes if n["references"])
    print(f"Nodes with §-refs: {with_refs} ({with_refs/len(nodes):.0%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    nodes = parse_corpus(limit=args.limit)

    if not nodes:
        print("No nodes parsed — check KB_ROOT path and keyword filter.", file=sys.stderr)
        sys.exit(1)

    print_stats(nodes)

    if not args.stats:
        OUTPUT_DIR.mkdir(exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(nodes, f, ensure_ascii=False, indent=2)
        print(f"\nWrote {len(nodes)} nodes to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
