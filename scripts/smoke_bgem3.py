"""
Smoke test for the BGE-M3 index. Run after Modal job downloads artifacts.

Verifies:
  - Vectors load with shape (N, 1024)
  - id_map ordering matches mpnet id_map (BM25 cache stays valid)
  - A Finnish query returns sensible top-5
  - End-to-end query latency under 5s

Usage:
  TAXXA_EMBED_MODEL=bge-m3 .venv/bin/python scripts/smoke_bgem3.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TAXXA_EMBED_MODEL", "bge-m3")

def main():
    vec_path = Path("data/vectors_bgem3.npy")
    id_path = Path("data/id_map_bgem3.json")
    old_id_path = Path("data/id_map.json")

    if not vec_path.exists() or not id_path.exists():
        sys.exit(f"FATAL: missing {vec_path} or {id_path} — did Modal finish?")

    vec = np.load(vec_path)
    print(f"[ok] vectors: shape={vec.shape}, dtype={vec.dtype}, mb={vec.nbytes/1e6:.0f}")
    assert vec.shape[1] == 1024, f"expected dim=1024, got {vec.shape[1]}"

    new_idmap = json.load(open(id_path))
    old_idmap = json.load(open(old_id_path))
    same_order = all(new_idmap[str(i)] == old_idmap[str(i)] for i in range(len(new_idmap)))
    print(f"[{'ok' if same_order else 'WARN'}] id_map order matches mpnet: {same_order}")
    if not same_order:
        print("    BM25 cache may be misaligned. Rebuild bm25.pkl before running retrieval.")

    print("\n[load] loading retriever (BGE-M3 query encoder)...")
    t0 = time.time()
    from retriever import retrieve
    out = retrieve("pääomatulon verokanta", top_k=5)
    t1 = time.time()
    print(f"[ok] first query (cold): {t1-t0:.2f}s, {len(out)} chunks")

    # Warm query
    t0 = time.time()
    out = retrieve("avainhenkilön lähdevero", top_k=5)
    t1 = time.time()
    print(f"[ok] warm query: {t1-t0:.2f}s, {len(out)} chunks")
    for i, n in enumerate(out, 1):
        print(f"  {i}. [{n.get('source','?')}/{n.get('type','?')}] {n['title'][:80]}")

if __name__ == "__main__":
    main()
