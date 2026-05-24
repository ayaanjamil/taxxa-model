"""
LLM-extracted typed edges for the Finnish tax law graph.

Today's graph has deterministic `cites` (Vero→Finlex via §-ref) edges. They are
correct as relationships, but they don't tell us *what kind* of relationship —
is the Vero doc interpreting the statute, clarifying a sub-case, citing a
court ruling that overrides it, or just mentioning it in passing? That
distinction is what lets graph-walk retrieval actually help on multi-hop QA.

For each existing `cites` edge, send the relevant chunks to an LLM and ask it
to re-type the relation. Resulting edges are written to
data/edge_extractions.jsonl. graph_builder.py loads them into a v2 graph
that is selectable via GRAPH_VERSION env var — the v1 deterministic graph
stays available as a fallback.

Costs are bounded:
  --sample N   run only N pairs (default 30 for dry-run validation)
  --full       run all unique pairs (~596 today; ~$0.60 on Haiku, less on DeepSeek)
  --resume     skip pairs already in edge_extractions.jsonl (cheap re-runs)
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

from openai import OpenAI

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
EXTRACT_MODEL = os.getenv("TAXXA_EXTRACT_MODEL", "anthropic/claude-haiku-4-5")

DATA_DIR = Path("data")
GRAPH_PKL = DATA_DIR / "graph.pkl"
NODES_PATH = DATA_DIR / "nodes.json"
OUT_PATH = DATA_DIR / "edge_extractions.jsonl"

# Cap how much of each parent's text we feed the extractor. Tax docs can be
# 50KB+; we only need the first chunk or two to read the type of relationship.
MAX_CHARS_PER_DOC = 2500

SYSTEM = """Olet suomalaisten veroasiakirjojen analyytikko (You are an analyst of Finnish tax documents).

You will be given two Finnish documents — document A (Verohallinto guidance) and document B (Finlex statute) — that are known to reference each other. Your job is to classify the *type* of relationship.

Return JSON: {"relation": "<one of: interpreted_by | clarified_by | overrides | references | none>", "confidence": <0.0-1.0>, "reasoning": "<one sentence in Finnish or English>"}

Definitions:
- "interpreted_by" — document A explains how to apply the statute B in practice (the most common case for Vero guidance). Confidence ≥ 0.8 typical.
- "clarified_by"  — document A adds a clarifying detail to a narrow sub-case of B (more specific than interpreted_by; example: clarifying which expenses qualify).
- "overrides"     — A is a court ruling (KHO/KKO) or later statute that *changes* the meaning of B. Rare for Vero→Finlex pairs.
- "references"    — A merely mentions B in passing without explaining or applying it.
- "none"          — the reference is incidental, generic, or the docs don't actually relate substantively.

Be strict. Default to "references" if you're not sure the relationship is interpretive.
Return only the JSON object, no markdown fences."""

EXAMPLES = """Example A (interpreted_by):
DOC A title: "Pääomatulojen verotus – Verohallinnon ohje"
DOC A excerpt: "Tämä ohje käsittelee tuloverolain (TVL) 32 §:n soveltamista pääomatulojen verotuksessa. Pääomatulosta peritään veroa 30 prosenttia 30 000 euroon asti..."
DOC B title: "Tuloverolaki 32 §"
DOC B excerpt: "Pääomatulosta suoritetaan veroa valtiolle. Veroprosentti on..."
Output: {"relation": "interpreted_by", "confidence": 0.92, "reasoning": "Vero ohje selittää eksplisiittisesti, miten TVL 32 § sovelletaan käytännössä, sisältäen veroprosentit."}

Example B (references):
DOC A title: "Yleinen verotusohje 2024 – Verohallinto"
DOC A excerpt: "...ks. myös tuloverolain 124 §..."
DOC B title: "Tuloverolaki 124 §"
DOC B excerpt: "Vero määrätään verotettavasta tulosta..."
Output: {"relation": "references", "confidence": 0.85, "reasoning": "Document A only mentions §124 in passing without interpreting or applying it."}"""


def _client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("ERROR: set OPENROUTER_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=key, base_url=OPENROUTER_BASE)


def _load_pairs() -> list[tuple[str, str]]:
    with open(GRAPH_PKL, "rb") as f:
        g = pickle.load(f)
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for u, v, d in g.edges(data=True):
        if d.get("relation") != "cites":
            continue
        if (u, v) in seen:
            continue
        seen.add((u, v))
        pairs.append((u, v))
    return pairs


def _parent_excerpt(node_list: list[dict], parent_id: str) -> tuple[str, str]:
    """Return (title, first-N-chars-of-text) for a parent. Reads chunks in order."""
    title = ""
    parts: list[str] = []
    total = 0
    for n in node_list:
        pid = n.get("parent_id") or n["id"].rsplit("#chunk", 1)[0]
        if pid != parent_id:
            continue
        if not title:
            title = n.get("title", "") or parent_id
        text = n.get("text") or ""
        if total + len(text) > MAX_CHARS_PER_DOC:
            parts.append(text[: MAX_CHARS_PER_DOC - total])
            break
        parts.append(text)
        total += len(text)
        if total >= MAX_CHARS_PER_DOC:
            break
    return title, "\n".join(parts)


def _extract_one(client: OpenAI, a_title: str, a_text: str, b_title: str, b_text: str) -> dict | None:
    user = (
        f"{EXAMPLES}\n\n"
        f"Now classify:\n"
        f"DOC A title: {a_title}\n"
        f"DOC A excerpt: {a_text}\n\n"
        f"DOC B title: {b_title}\n"
        f"DOC B excerpt: {b_text}"
    )
    try:
        resp = client.chat.completions.create(
            model=EXTRACT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            max_tokens=200,
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return json.loads(raw)
    except Exception as e:
        return {"relation": "error", "confidence": 0.0, "reasoning": str(e)[:120]}


def _load_done() -> set[tuple[str, str]]:
    """Pairs already in the output file — skipped on --resume."""
    done: set[tuple[str, str]] = set()
    if not OUT_PATH.exists():
        return done
    with open(OUT_PATH) as f:
        for line in f:
            try:
                row = json.loads(line)
                done.add((row["from"], row["to"]))
            except Exception:
                continue
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=30, help="Run N random pairs (default 30)")
    ap.add_argument("--full", action="store_true", help="Run all pairs (~$0.60 on Haiku)")
    ap.add_argument("--resume", action="store_true", help="Skip pairs already in edge_extractions.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading pairs from {GRAPH_PKL}...")
    pairs = _load_pairs()
    print(f"  {len(pairs)} unique Vero→Finlex pairs")

    print(f"Loading {NODES_PATH}...")
    with open(NODES_PATH, encoding="utf-8") as f:
        node_list = json.load(f)
    print(f"  {len(node_list)} chunks")

    done = _load_done() if args.resume else set()
    if done:
        print(f"  resuming — {len(done)} pairs already extracted")
        pairs = [p for p in pairs if p not in done]

    if not args.full:
        random.seed(args.seed)
        pairs = random.sample(pairs, min(args.sample, len(pairs)))
        print(f"  --sample {len(pairs)} pairs")
    else:
        print(f"  --full {len(pairs)} pairs (estimated cost: ~${len(pairs) * 0.001:.2f} on Haiku)")

    client = _client()
    t0 = time.perf_counter()
    by_rel: dict[str, int] = {}

    with open(OUT_PATH, "a", encoding="utf-8") as out:
        for i, (a, b) in enumerate(pairs):
            a_title, a_text = _parent_excerpt(node_list, a)
            b_title, b_text = _parent_excerpt(node_list, b)
            if not a_text or not b_text:
                continue
            result = _extract_one(client, a_title, a_text, b_title, b_text)
            if not result:
                continue
            rel = result.get("relation", "error")
            by_rel[rel] = by_rel.get(rel, 0) + 1
            row = {
                "from": a, "to": b,
                "relation": rel,
                "confidence": result.get("confidence", 0.0),
                "reasoning": result.get("reasoning", ""),
                "model": EXTRACT_MODEL,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  [{i+1}/{len(pairs)}] {elapsed:.1f}s — by_rel={by_rel}")

    print(f"\nDone. {sum(by_rel.values())} edges classified in {time.perf_counter() - t0:.1f}s")
    print(f"  by relation: {by_rel}")
    print(f"  output: {OUT_PATH}")


if __name__ == "__main__":
    main()
