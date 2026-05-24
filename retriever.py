"""
Hybrid retrieval: vector (multilingual-mpnet) + BM25 over full chunk text,
merged with Reciprocal Rank Fusion. Lazy-loaded singletons.

Why hybrid: the embedding model semantically clusters all pension docs together,
so "TyEL" queries match VEL / lisäeläkevakuutus / eläketulon verotus equally well.
BM25 on the chunk's full text catches exact matches ("TyEL", "syntymävuosi",
"68 vuoden iässä") even when vector similarity smears them across the corpus.

Query expansion: Finnish legal acronyms (TyEL, TVL, AVL...) are opaque to the
embedder. We expand them to full form before encoding so vector search has
real semantic surface to match.

§-reference expansion: retained from the previous design — if a top-ranked
chunk cites "§ 4", we surface other chunks of that same statute's §4.
"""

import json
import os
import pickle
import re
from pathlib import Path

import libvoikko
import networkx as nx
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(os.getenv("TAXXA_DATA_DIR", "data"))
BM25_CACHE = DATA_DIR / "bm25.pkl"
VOIKKO_CACHE = DATA_DIR / "voikko_cache.json"

# GRAPH_VERSION: "v1" (deterministic only, default) | "v2" (with LLM-typed edges)
_GRAPH_VERSION = os.getenv("TAXXA_GRAPH_VERSION", "v1")
_GRAPH_FILE = {"v1": "graph.pkl", "v2": "graph_v2.pkl"}.get(_GRAPH_VERSION, "graph.pkl")
GRAPH_PKL = DATA_DIR / _GRAPH_FILE

# Match embedder.py's registry so index and query encoder are always paired.
_EMBED_MODELS = {
    "mpnet":  ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 256,
               str(DATA_DIR / "vectors.npy"),       str(DATA_DIR / "id_map.json")),
    "bge-m3": ("BAAI/bge-m3", 512,
               str(DATA_DIR / "vectors_bgem3.npy"), str(DATA_DIR / "id_map_bgem3.json")),
}
_EMBED_KEY = os.getenv("TAXXA_EMBED_MODEL", "mpnet")
MODEL_NAME, _MAX_SEQ, VECTORS_PATH, ID_MAP_PATH = _EMBED_MODELS[_EMBED_KEY]
# Env-var overrides (handy for ablations without touching the registry)
VECTORS_PATH = os.getenv("TAXXA_VECTORS_FILE", VECTORS_PATH)
ID_MAP_PATH  = os.getenv("TAXXA_ID_MAP_FILE", ID_MAP_PATH)

# Retrieval-tuning A/B switches. Read once at import; eval scripts set them in env.
# DEDUP_MODE: "post" (today, dedup AFTER rank) | "pre" (dedup before RRF)
# RRF_K: 60 (today's hard-default) | 100 (softer penalty on mid-rank candidates)
DEDUP_MODE_DEFAULT = os.getenv("TAXXA_DEDUP_MODE", "post")
RRF_K_DEFAULT = int(os.getenv("TAXXA_RRF_K", "60"))

# Finnish legal acronyms — expanded into queries before embedding/tokenizing
ACRONYMS = {
    "TyEL": "Työntekijän eläkelaki (TyEL)",
    "TVL":  "Tuloverolaki (TVL)",
    "AVL":  "Arvonlisäverolaki (AVL)",
    "MEL":  "Merimieseläkelaki (MEL)",
    "YEL":  "Yrittäjän eläkelaki (YEL)",
    "MYEL": "Maatalousyrittäjien eläkelaki (MYEL)",
    "EPL":  "Ennakkoperintälaki (EPL)",
    "VML":  "Laki verotusmenettelystä (VML)",
    "PerVL": "Perintö- ja lahjaverolaki (PerVL)",
    "KiVL":  "Kiinteistöverolaki (KiVL)",
    "EVL":   "Elinkeinoverolaki (EVL)",
}

_model: SentenceTransformer | None = None
_vectors: np.ndarray | None = None
_id_map: dict[int, str] | None = None
_nid_to_idx: dict[str, int] | None = None
_nodes: dict[str, dict] | None = None
_node_list: list[dict] | None = None      # ordered by index — parallel to _vectors
_section_index: dict[str, list[str]] | None = None
_bm25: BM25Okapi | None = None
_graph: nx.DiGraph | None = None
_parent_to_chunks: dict[str, list[str]] | None = None

_SEC_RE = re.compile(r"(\d+[a-zA-Z]?)\s*§|§\s*(\d+[a-zA-Z]?)")
_TOKEN_RE = re.compile(r"[a-zäöå0-9§]+", re.UNICODE)

_voikko: libvoikko.Voikko | None = None
_voikko_cache: dict[str, str] = {}  # token → lowercased baseform


def _get_voikko() -> libvoikko.Voikko:
    global _voikko
    if _voikko is None:
        _voikko = libvoikko.Voikko("fi")
    return _voikko


def _baseform(token: str) -> str:
    """Return Voikko base form for a Finnish token (lowercased). Falls back to original."""
    if token in _voikko_cache:
        return _voikko_cache[token]
    try:
        analyses = _get_voikko().analyze(token)
        bf = analyses[0]["BASEFORM"].lower() if analyses else token
    except Exception:
        bf = token
    _voikko_cache[token] = bf
    return bf


def _tokenize_fi(s: str) -> list[str]:
    tokens = _TOKEN_RE.findall(s.lower())
    result = []
    for t in tokens:
        result.append(t)
        bf = _baseform(t)
        if bf != t:
            result.append(bf)
    return result


def expand_query(q: str) -> str:
    """Expand Finnish legal acronyms in-place. Case-sensitive on the acronym."""
    for short, long in ACRONYMS.items():
        # Replace only standalone acronym tokens (avoid mangling URLs etc.)
        q = re.sub(rf"(?<![A-Za-z]){re.escape(short)}(?![A-Za-z])", long, q)
    return q


def _build_section_index(nodes: dict[str, dict]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for nid, n in nodes.items():
        statute = n.get("statute")
        if not statute:
            continue
        combined = n.get("title", "") + " " + n.get("text", "")[:1500]
        for m in _SEC_RE.finditer(combined):
            sec = m.group(1) or m.group(2)
            if sec:
                key = f"{statute}_§{sec}"
                index.setdefault(key, [])
                if nid not in index[key]:
                    index[key].append(nid)
    return index


def _load():
    global _model, _vectors, _id_map, _nid_to_idx, _nodes, _node_list
    global _section_index, _bm25

    if _model is None:
        # MPS gives ~3-5x speedup on Apple Silicon vs CPU for BGE-M3 query encoding;
        # falls back to CPU silently elsewhere.
        import torch
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading embedding model: {MODEL_NAME} (device={device})")
        _model = SentenceTransformer(MODEL_NAME, device=device)
        _model.max_seq_length = _MAX_SEQ

    if _vectors is None:
        print(f"Loading vectors from {VECTORS_PATH}...")
        _vectors = np.load(VECTORS_PATH)

    if _id_map is None:
        with open(ID_MAP_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        _id_map = {int(k): v for k, v in raw.items()}
        _nid_to_idx = {v: int(k) for k, v in raw.items()}

    if _nodes is None:
        print("Loading nodes...")
        with open(DATA_DIR / "nodes.json", encoding="utf-8") as f:
            node_list = json.load(f)
        _nodes = {n["id"]: n for n in node_list}
        # Build parallel-to-vectors list for BM25 index alignment
        _node_list = [_nodes[_id_map[i]] for i in range(len(_id_map))]

    if _section_index is None:
        _section_index = _build_section_index(_nodes)

    # Load Voikko token cache before any tokenization happens
    if VOIKKO_CACHE.exists() and not _voikko_cache:
        with open(VOIKKO_CACHE, encoding="utf-8") as f:
            _voikko_cache.update(json.load(f))
        print(f"Loaded Voikko cache: {len(_voikko_cache)} tokens")

    if _bm25 is None:
        if BM25_CACHE.exists():
            print("Loading BM25 from cache...")
            with open(BM25_CACHE, "rb") as f:
                _bm25 = pickle.load(f)
            print(f"BM25 loaded from cache.")
        else:
            print(f"Building BM25 index over {len(_node_list)} chunks...")
            tokenized = [
                _tokenize_fi(n.get("title", "") + " " + n.get("text", ""))
                for n in _node_list
            ]
            _bm25 = BM25Okapi(tokenized)
            print("BM25 built. Saving to cache...")
            with open(BM25_CACHE, "wb") as f:
                pickle.dump(_bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(VOIKKO_CACHE, "w", encoding="utf-8") as f:
                json.dump(_voikko_cache, f, ensure_ascii=False)
            print(f"Cached BM25 and {len(_voikko_cache)} Voikko tokens.")


def _rrf_merge(rank_lists: list[dict[int, int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion. rank_lists is a list of {idx: rank} dicts."""
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for idx, rank in ranks.items():
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores


def _parent_of(nid: str) -> str:
    n = (_nodes or {}).get(nid)
    if n and n.get("parent_id"):
        return n["parent_id"]
    return nid.rsplit("#chunk", 1)[0] if "#chunk" in nid else nid


def _predupe_ranked_idxs(idxs: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Keep, for each parent, only the highest-scoring chunk index from a ranked list.

    Order in `idxs` is already by descending score, so the first time we see a
    parent is its best chunk. Returns idxs filtered in-place; preserves order.
    """
    seen: set[str] = set()
    keep: list[int] = []
    for idx in idxs:
        pid = _parent_of(_id_map[int(idx)])
        if pid in seen:
            continue
        seen.add(pid)
        keep.append(int(idx))
    return np.array(keep, dtype=idxs.dtype) if keep else idxs


def retrieve(
    query: str,
    top_k: int = 10,
    candidate_pool: int = 50,
    dedup_mode: str | None = None,
    rrf_k: int | None = None,
) -> list[dict]:
    """Hybrid retrieve: vector + BM25 → RRF merge → §-ref expansion → ranked top-k."""
    _load()

    if dedup_mode is None:
        dedup_mode = DEDUP_MODE_DEFAULT
    if rrf_k is None:
        rrf_k = RRF_K_DEFAULT

    expanded = expand_query(query)

    # When dedup_mode="pre" we collect a wider candidate pool from each list,
    # then collapse to top-N unique parents BEFORE RRF — so two chunks of the
    # same parent don't both eat slots in the merge.
    if dedup_mode == "pre":
        wide_pool = max(candidate_pool * 3, candidate_pool)
        target_unique = min(candidate_pool, 40)
    else:
        wide_pool = candidate_pool
        target_unique = candidate_pool

    # --- Vector scores
    q_vec = _model.encode(expanded, normalize_embeddings=True)
    vec_scores = _vectors @ q_vec  # (N,)
    vec_top = np.argpartition(-vec_scores, wide_pool)[:wide_pool]
    vec_top = vec_top[np.argsort(-vec_scores[vec_top])]
    if dedup_mode == "pre":
        vec_top = _predupe_ranked_idxs(vec_top, vec_scores)[:target_unique]
    vec_rank = {int(idx): rank for rank, idx in enumerate(vec_top)}

    # --- BM25 scores
    bm25_scores = _bm25.get_scores(_tokenize_fi(expanded))
    bm25_top = np.argpartition(-bm25_scores, wide_pool)[:wide_pool]
    bm25_top = bm25_top[np.argsort(-bm25_scores[bm25_top])]
    if dedup_mode == "pre":
        bm25_top = _predupe_ranked_idxs(bm25_top, bm25_scores)[:target_unique]
    bm25_rank = {int(idx): rank for rank, idx in enumerate(bm25_top)}

    # --- RRF merge
    fused = _rrf_merge([vec_rank, bm25_rank], k=rrf_k)

    # Vero authority boost: prefer current Verohallinto guidance
    # Treaty boost: bilateral tax treaties (Tuloverosopimukset) are primary law —
    # give them the same boost so the vero preference doesn't bury them
    for idx, score in list(fused.items()):
        nid = _id_map[idx]
        if _nodes.get(nid, {}).get("source") == "vero":
            fused[idx] = score * 1.15
        elif "Tuloverosopimukset" in nid:
            fused[idx] = score * 1.15

    # --- §-reference expansion + treaty neighbor expansion
    expanded_scores: dict[str, float] = {}
    for idx, score in fused.items():
        nid = _id_map[idx]
        expanded_scores[nid] = max(expanded_scores.get(nid, 0.0), score)
        node = _nodes.get(nid, {})
        for ref_key in node.get("references", [])[:3]:
            ref_score = score * 0.6
            for resolved_id in _section_index.get(ref_key, [])[:2]:
                if resolved_id not in expanded_scores:
                    expanded_scores[resolved_id] = ref_score
        # Treaty neighbor expansion: bilateral treaty docs have no §-references, so
        # pull in adjacent chunks (±3) when a treaty chunk is retrieved. This surfaces
        # article-level content (e.g. Article 10 dividend rates) near a matched preamble.
        if "Tuloverosopimukset" in nid and "#chunk" in nid:
            try:
                base, chunk_str = nid.rsplit("#chunk", 1)
                chunk_idx = int(chunk_str)
                for offset in range(-3, 4):
                    if offset == 0:
                        continue
                    neighbor_id = f"{base}#chunk{chunk_idx + offset}"
                    if neighbor_id in _nodes and neighbor_id not in expanded_scores:
                        expanded_scores[neighbor_id] = score * 0.75
            except (ValueError, IndexError):
                pass

    # --- Sort: non-superseded first, then score
    ranked = sorted(
        expanded_scores.items(),
        key=lambda kv: (
            1 if _nodes.get(kv[0], {}).get("superseded_by") else 0,
            -kv[1],
        ),
    )

    # Cap chunks per parent doc so top_k surfaces distinct sources.
    # Treaty docs are long and structured — allow more chunks from them
    # so individual articles (dividend, interest, PE...) can all surface.
    from collections import Counter
    out: list[dict] = []
    parent_count: Counter[str] = Counter()
    for nid, score in ranked:
        node = _nodes.get(nid)
        if not node:
            continue
        parent = node.get("parent_id") or node["id"]
        max_per_parent = 4 if "Tuloverosopimukset" in nid else 2
        if parent_count[parent] >= max_per_parent:
            continue
        parent_count[parent] += 1
        # Shallow copy + attach retrieval metadata. Avoids mutating the shared
        # _nodes dict; callers that want the metadata read _retrieval_score /
        # _retrieval_rank, callers that don't simply ignore them.
        out.append({**node, "_retrieval_score": float(score), "_retrieval_rank": len(out) + 1})
        if len(out) >= top_k:
            break
    return out


def _load_graph() -> None:
    """Lazy-load the parent-level graph + parent→chunks mapping."""
    global _graph, _parent_to_chunks
    if _graph is not None:
        return
    _load()  # need _nodes for parent_to_chunks
    if not GRAPH_PKL.exists():
        print(f"WARNING: {GRAPH_PKL} not found — graph traversal disabled")
        _graph = nx.DiGraph()
        _parent_to_chunks = {}
        return
    print(f"Loading graph ({_GRAPH_VERSION}: {GRAPH_PKL.name})...")
    with open(GRAPH_PKL, "rb") as f:
        _graph = pickle.load(f)
    _parent_to_chunks = {}
    for n in _node_list or []:
        pid = n.get("parent_id") or (n["id"].rsplit("#chunk", 1)[0] if "#chunk" in n["id"] else n["id"])
        _parent_to_chunks.setdefault(pid, []).append(n["id"])
    print(f"Graph loaded: {_graph.number_of_nodes()} nodes, {_graph.number_of_edges()} edges")


_TYPE_PRIORITY = {"ARTICLE": 0, "CLAUSE": 0, "STATUTE": 0,
                  "GUIDANCE_S": 1, "GUIDANCE": 1,
                  "COURT_CASE": 2}


def retrieve_with_graph(
    query: str,
    top_k: int = 10,
    candidate_pool: int = 50,
    max_hops: int = 2,
    hop_relations: tuple[str, ...] = ("cites", "amends", "interpreted_by", "overrides"),
    frontier_cap: int = 20,
) -> tuple[list[dict], list[dict]]:
    """Graph-walk retrieval. Returns (chunks, traversal_log).

    traversal_log entries: {"from": parent_id, "to": parent_id, "relation": str}
    """
    _load_graph()

    # Get entry chunks at higher k so we have headroom after graph expansion.
    # Widened from top_k*2 to top_k*3 — the previous 20-chunk frontier was too
    # narrow on hard tier (right parent often sat at rank 21-30).
    entry_frontier_mult = int(os.getenv("TAXXA_ENTRY_FRONTIER_MULT", "3"))
    entry_chunks = retrieve(
        query,
        top_k=top_k * entry_frontier_mult,
        candidate_pool=candidate_pool,
    )
    entry_parents: list[str] = []
    seen_parents: set[str] = set()
    # Map entry parent → ordered list of chunks BM25/vector actually matched
    entry_parent_chunks: dict[str, list[dict]] = {}
    for n in entry_chunks:
        pid = n.get("parent_id") or (n["id"].rsplit("#chunk", 1)[0] if "#chunk" in n["id"] else n["id"])
        entry_parent_chunks.setdefault(pid, []).append(n)
        if pid not in seen_parents:
            seen_parents.add(pid)
            entry_parents.append(pid)

    neighborhood: set[str] = set(entry_parents)
    traversal_log: list[dict] = []
    frontier = list(entry_parents)

    for _hop in range(max_hops):
        next_frontier: list[str] = []
        for pid in frontier:
            if pid not in _graph:
                continue
            for neighbor in _graph.successors(pid):
                rel = _graph[pid][neighbor].get("relation", "references")
                if rel in hop_relations and neighbor not in neighborhood:
                    traversal_log.append({"from": pid, "to": neighbor, "relation": rel})
                    neighborhood.add(neighbor)
                    next_frontier.append(neighbor)
            for pred in _graph.predecessors(pid):
                rel = _graph[pred][pid].get("relation", "")
                if rel == "amends" and pred not in neighborhood:
                    traversal_log.append({"from": pred, "to": pid, "relation": "amends (newer)"})
                    neighborhood.add(pred)
                    next_frontier.append(pred)
            if len(next_frontier) >= frontier_cap:
                break
        frontier = next_frontier[:frontier_cap]
        if not frontier:
            break

    # Rank parents: entry parents first (preserving vector+BM25 order),
    # then graph-expanded parents by type priority and date
    def _parent_sort_key(pid: str) -> tuple:
        chunks = (_parent_to_chunks or {}).get(pid) or []
        rep = (_nodes or {}).get(chunks[0], {}) if chunks else {}
        type_rank = _TYPE_PRIORITY.get(rep.get("type") or "", 3)
        date_str = rep.get("date") or "0000-00-00"
        return (type_rank, -ord(date_str[0]) if date_str else 0, date_str)

    entry_set = set(entry_parents)
    expanded_only = sorted([p for p in neighborhood if p not in entry_set], key=_parent_sort_key)
    ranked_parents = entry_parents + expanded_only

    # Expand parents → chunks, apply parent-cap.
    # For entry parents, use BM25-ranked chunks (preserves the specific match);
    # for graph-expanded parents, fall back to natural chunk order.
    from collections import Counter
    out: list[dict] = []
    parent_count: Counter[str] = Counter()
    for pid in ranked_parents:
        max_per_parent = 4 if "Tuloverosopimukset" in pid else 2
        chunks_in_order: list[dict] = []
        # Preferred: chunks already matched by BM25/vector
        for n in entry_parent_chunks.get(pid, []):
            chunks_in_order.append(n)
        # Fallback / additional: natural chunk order (for graph-expanded parents)
        seen_ids = {n["id"] for n in chunks_in_order}
        for cid in (_parent_to_chunks or {}).get(pid, []):
            if cid in seen_ids:
                continue
            node = (_nodes or {}).get(cid)
            if node:
                chunks_in_order.append(node)
        for node in chunks_in_order:
            if parent_count[pid] >= max_per_parent:
                break
            parent_count[pid] += 1
            out.append(node)
        if len(out) >= top_k:
            break
    return out[:top_k], traversal_log


def format_nodes(nodes: list[dict], max_chars: int = 8000) -> str:
    parts = []
    total = 0
    for n in nodes:
        header = f"[{n['id']}] {n['title']} ({n.get('source','?')}, {n.get('type','?')})"
        if n.get("date"):
            header += f" — {n['date']}"
        body = n["text"]
        block = f"{header}\n{body}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)


def diagnose(query: str, expected_substring: str, top_k: int = 30) -> int | None:
    """Did a chunk containing `expected_substring` land in top-k? Returns rank or None."""
    results = retrieve(query, top_k=top_k)
    for rank, n in enumerate(results, 1):
        if expected_substring.lower() in n["text"].lower():
            print(f"  ✓ Found at rank {rank}: {n['id'][:90]}")
            return rank
    print(f"  ✗ Not in top {top_k}. Top 5 instead:")
    for rank, n in enumerate(results[:5], 1):
        print(f"    {rank}. {n['id'][:90]}")
    return None
