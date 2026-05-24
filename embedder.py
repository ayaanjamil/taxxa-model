"""
Embed nodes from data/nodes.json using a multilingual sentence-transformer.
Writes data/vectors.npy and data/id_map.json.

Usage:
  python embedder.py                  # embed all nodes
  python embedder.py --sample 500     # embed N nodes (fast dev/test)
  python embedder.py --batch-size 64  # tune for your RAM
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

NODES_FILE = Path("data/nodes.json")

# Two supported embedders. Selectable via env or --model:
#   mpnet (default, legacy 768-dim) | bge-m3 (1024-dim, much stronger on Finnish)
MODELS = {
    "mpnet":  ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 256,
               "data/vectors.npy",        "data/id_map.json"),
    "bge-m3": ("BAAI/bge-m3", 512,
               "data/vectors_bgem3.npy",  "data/id_map_bgem3.json"),
}

MODEL_KEY_DEFAULT = os.getenv("TAXXA_EMBED_MODEL", "mpnet")
MODEL_NAME, MAX_SEQ_LENGTH, VECTORS_FILE, ID_MAP_FILE = MODELS[MODEL_KEY_DEFAULT]
VECTORS_FILE = Path(VECTORS_FILE)
ID_MAP_FILE = Path(ID_MAP_FILE)


def load_nodes(sample: int | None = None) -> list[dict]:
    with open(NODES_FILE, encoding="utf-8") as f:
        nodes = json.load(f)
    if sample:
        nodes = random.sample(nodes, min(sample, len(nodes)))
    return nodes


def embed_nodes(nodes: list[dict], batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH

    # Each node is already chunked (~800 chars) by parser.chunk_text — embed full chunk.
    # Prefix with title so the embedding carries document-level context.
    texts = [f"{n['title']}\n\n{n['text']}" for n in nodes]

    print(f"Embedding {len(texts)} nodes (batch_size={batch_size})...")
    t0 = time.time()

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # enables dot-product as cosine similarity
    )

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s — {len(vectors)} vectors, dim={vectors.shape[1]}")
    return vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if not NODES_FILE.exists():
        print(f"ERROR: {NODES_FILE} not found. Run parser.py first.", file=sys.stderr)
        sys.exit(1)

    nodes = load_nodes(sample=args.sample)
    print(f"Loaded {len(nodes)} nodes from {NODES_FILE}")

    vectors = embed_nodes(nodes, batch_size=args.batch_size)

    Path("data").mkdir(exist_ok=True)
    np.save(VECTORS_FILE, vectors)

    id_map = {i: n["id"] for i, n in enumerate(nodes)}
    with open(ID_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False)

    mb = vectors.nbytes / 1024 / 1024
    print(f"\nSaved {VECTORS_FILE} ({mb:.1f} MB) and {ID_MAP_FILE}")
    print(f"Shape: {vectors.shape}")


if __name__ == "__main__":
    main()
