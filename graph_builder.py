"""
Build a typed legal-document graph from nodes.json.

Operates on PARENT documents (not chunks): collapse chunks into parents, then
build deterministic edges from existing node fields — no LLM required for the
core graph. LLM classification is optional (off by default).

Edge types:
  - "cites":  vero parent → finlex parent(s) that define a cited §
  - "amends": finlex amendment parent → finlex base-law parent(s) for the amended §

Output:
  data/graph.pkl         — pickled networkx DiGraph (nodes = parent IDs)
  data/graph_edges.json  — edge list for debugging / frontend
  data/parent_nodes.json — parent registry (id → metadata + chunk_ids)
"""

import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx

DATA_DIR = Path("data")
NODES_PATH = DATA_DIR / "nodes.json"
GRAPH_PKL = DATA_DIR / "graph.pkl"
GRAPH_V2_PKL = DATA_DIR / "graph_v2.pkl"
EDGES_JSON = DATA_DIR / "graph_edges.json"
PARENTS_JSON = DATA_DIR / "parent_nodes.json"
EDGE_EXTRACTIONS = DATA_DIR / "edge_extractions.jsonl"
LLM_CONF_THRESHOLD = 0.7  # confidence floor for accepting an LLM relation re-typing

# Match "12 §" or "§ 12" or "12 a §" — same as retriever
_SEC_RE = re.compile(r"(\d+[a-zA-Z]?)\s*§|§\s*(\d+[a-zA-Z]?)")
# Match "69 ja 71 §" — Finnish amendment titles list multiple sections
_SEC_JA_RE = re.compile(r"(\d+[a-zA-Z]?)\s+ja\s+(\d+[a-zA-Z]?)\s*§")
_AMENDS_RE = re.compile(r"muuttamisesta", re.IGNORECASE)

MAX_CITES_PER_REF = 3  # cap fan-out per reference
MAX_AMENDS_TARGETS = 3


def _strip_chunk(node_id: str) -> str:
    """Strip the #chunkN suffix to get the parent_id."""
    if "#chunk" in node_id:
        return node_id.rsplit("#chunk", 1)[0]
    return node_id


def build_parent_nodes(node_list: list[dict]) -> dict[str, dict]:
    """Collapse chunks into parent documents."""
    parents: dict[str, dict] = {}
    for n in node_list:
        pid = n.get("parent_id") or _strip_chunk(n["id"])
        if pid not in parents:
            parents[pid] = {
                "id": pid,
                "source": n.get("source"),
                "type": n.get("type"),
                "statute": n.get("statute"),
                "title": n.get("title", ""),
                "date": n.get("date"),
                "chunk_ids": [],
                "references": [],
            }
        p = parents[pid]
        p["chunk_ids"].append(n["id"])
        for ref in n.get("references") or []:
            if ref not in p["references"]:
                p["references"].append(ref)
        # First non-null date wins
        if not p["date"] and n.get("date"):
            p["date"] = n["date"]
    return parents


def build_section_index(parents: dict[str, dict]) -> dict[str, list[str]]:
    """Map 'STATUTE_§N' → [parent_id, ...].

    Sources:
      1. Each parent's references list (already in STATUTE_§N form).
      2. Section numbers detected in the parent's title (for amendment laws like
         "Laki tuloverolain 38 §:n muuttamisesta") — index under the statute.
    """
    index: dict[str, list[str]] = defaultdict(list)

    for pid, p in parents.items():
        for ref_key in p["references"]:
            if pid not in index[ref_key]:
                index[ref_key].append(pid)

        # Also extract sections mentioned in the title and index under this statute
        statute = p.get("statute")
        if statute:
            title = p.get("title", "")
            section_nums: set[str] = set()
            for m in _SEC_RE.finditer(title):
                section_nums.add(m.group(1) or m.group(2))
            # Catch Finnish "X ja Y §" patterns where only Y has the § immediately
            for m in _SEC_JA_RE.finditer(title):
                section_nums.add(m.group(1))
                section_nums.add(m.group(2))
            for sec in section_nums:
                key = f"{statute}_§{sec}"
                if pid not in index[key]:
                    index[key].append(pid)

    return dict(index)


def build_edges(
    parents: dict[str, dict],
    section_index: dict[str, list[str]],
) -> list[dict]:
    """Build typed edges from parent metadata. No LLM."""
    edges: list[dict] = []

    # --- "cites": vero → finlex (resolved via section_index)
    for pid, p in parents.items():
        if p["source"] != "vero":
            continue
        for ref_key in p["references"]:
            targets = section_index.get(ref_key, [])
            added = 0
            for target_pid in targets:
                if target_pid == pid:
                    continue
                target = parents.get(target_pid, {})
                if target.get("source") != "finlex":
                    continue
                edges.append({
                    "from": pid,
                    "to": target_pid,
                    "relation": "cites",
                    "confidence": 0.85,
                    "ref_key": ref_key,
                })
                added += 1
                if added >= MAX_CITES_PER_REF:
                    break

    # --- "amends": finlex amendment → finlex base law
    for pid, p in parents.items():
        if p["source"] != "finlex":
            continue
        title = p.get("title", "") or ""
        if not _AMENDS_RE.search(title):
            continue
        statute = p.get("statute")
        if not statute:
            continue

        section_nums: set[str] = set()
        for m in _SEC_RE.finditer(title):
            section_nums.add(m.group(1) or m.group(2))
        for m in _SEC_JA_RE.finditer(title):
            section_nums.add(m.group(1))
            section_nums.add(m.group(2))

        for sec in section_nums:
            ref_key = f"{statute}_§{sec}"
            added = 0
            for target_pid in section_index.get(ref_key, []):
                if target_pid == pid:
                    continue
                target = parents.get(target_pid, {})
                if target.get("source") != "finlex":
                    continue
                # Don't link amendment-to-amendment
                if _AMENDS_RE.search(target.get("title", "") or ""):
                    continue
                edges.append({
                    "from": pid,
                    "to": target_pid,
                    "relation": "amends",
                    "confidence": 0.75,
                    "ref_key": ref_key,
                })
                added += 1
                if added >= MAX_AMENDS_TARGETS:
                    break

    return edges


def assemble_graph(parents: dict[str, dict], edges: list[dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    for pid, p in parents.items():
        G.add_node(
            pid,
            source=p.get("source"),
            type=p.get("type"),
            statute=p.get("statute"),
            title=p.get("title"),
            date=p.get("date"),
            chunk_count=len(p["chunk_ids"]),
        )
    for e in edges:
        G.add_edge(
            e["from"], e["to"],
            relation=e["relation"],
            confidence=e["confidence"],
            ref_key=e.get("ref_key"),
        )
    return G


def build_v2(base_graph: nx.DiGraph) -> tuple[nx.DiGraph, dict[str, int]]:
    """Layer LLM-extracted relations onto the v1 graph.

    For each (from, to) pair in edge_extractions.jsonl:
      - relation == "none" with confidence >= threshold  → remove the edge entirely
        (the deterministic §-ref was incidental, not substantive)
      - relation in {"interpreted_by", "clarified_by", "overrides"} → re-type the
        existing "cites" edge with the more specific relation, and overwrite confidence
      - relation == "references" → keep the "cites" edge but flag confidence lower
        (it's a real mention, but not interpretive — useful for graph viz, weak for retrieval)
      - confidence < threshold → leave the edge as it was in v1 (no change)

    Returns (v2_graph, stats).
    """
    G = base_graph.copy()
    stats: dict[str, int] = {"dropped_none": 0, "retyped": 0, "kept_references": 0, "low_conf_skipped": 0, "missing_edge": 0}

    if not EDGE_EXTRACTIONS.exists():
        print(f"WARNING: {EDGE_EXTRACTIONS} not found — v2 == v1")
        return G, stats

    with open(EDGE_EXTRACTIONS, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            u, v = row.get("from"), row.get("to")
            rel = row.get("relation")
            conf = float(row.get("confidence") or 0.0)
            if not (u and v and rel):
                continue
            if not G.has_edge(u, v):
                stats["missing_edge"] += 1
                continue
            if conf < LLM_CONF_THRESHOLD:
                stats["low_conf_skipped"] += 1
                continue
            if rel == "none":
                G.remove_edge(u, v)
                stats["dropped_none"] += 1
            elif rel in {"interpreted_by", "clarified_by", "overrides"}:
                G[u][v]["relation"] = rel
                G[u][v]["confidence"] = conf
                G[u][v]["llm_typed"] = True
                stats["retyped"] += 1
            elif rel == "references":
                G[u][v]["confidence"] = min(G[u][v].get("confidence", 1.0), 0.6)
                G[u][v]["llm_typed"] = True
                stats["kept_references"] += 1
    return G, stats


def main(stats_only: bool = False) -> None:
    print(f"Loading {NODES_PATH}...")
    with open(NODES_PATH, encoding="utf-8") as f:
        node_list = json.load(f)
    print(f"  {len(node_list)} chunks")

    print("Building parent registry...")
    parents = build_parent_nodes(node_list)
    print(f"  {len(parents)} parent documents")

    print("Building section index...")
    section_index = build_section_index(parents)
    print(f"  {len(section_index)} unique §-references indexed")

    print("Building edges...")
    edges = build_edges(parents, section_index)
    by_rel: dict[str, int] = defaultdict(int)
    for e in edges:
        by_rel[e["relation"]] += 1
    print(f"  {len(edges)} edges:")
    for rel, cnt in by_rel.items():
        print(f"    {rel}: {cnt}")

    if stats_only:
        return

    print("Assembling DiGraph...")
    G = assemble_graph(parents, edges)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print(f"Saving {GRAPH_PKL}...")
    with open(GRAPH_PKL, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saving {EDGES_JSON}...")
    with open(EDGES_JSON, "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)

    # --- v2: layer LLM-extracted relations on top
    if EDGE_EXTRACTIONS.exists():
        print(f"Building v2 from {EDGE_EXTRACTIONS}...")
        G_v2, v2_stats = build_v2(G)
        print(f"  v2 stats: {v2_stats}")
        print(f"  v2 graph: {G_v2.number_of_nodes()} nodes, {G_v2.number_of_edges()} edges")
        print(f"Saving {GRAPH_V2_PKL}...")
        with open(GRAPH_V2_PKL, "wb") as f:
            pickle.dump(G_v2, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(f"  (no {EDGE_EXTRACTIONS} — skipping v2)")

    print(f"Saving {PARENTS_JSON}...")
    # Strip chunk_ids before saving — they're huge and the retriever rebuilds the mapping
    parents_slim = {
        pid: {k: v for k, v in p.items() if k != "chunk_ids"}
        for pid, p in parents.items()
    }
    with open(PARENTS_JSON, "w", encoding="utf-8") as f:
        json.dump(parents_slim, f, ensure_ascii=False)

    print("Done.")


def main_v2_only() -> None:
    """Layer LLM edges onto the existing graph.pkl WITHOUT rebuilding from nodes.json."""
    print(f"Loading existing {GRAPH_PKL}...")
    with open(GRAPH_PKL, "rb") as f:
        G = pickle.load(f)
    print(f"  v1: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    G_v2, stats = build_v2(G)
    print(f"  stats: {stats}")
    print(f"  v2: {G_v2.number_of_nodes()} nodes, {G_v2.number_of_edges()} edges")
    print(f"Saving {GRAPH_V2_PKL}...")
    with open(GRAPH_V2_PKL, "wb") as f:
        pickle.dump(G_v2, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done.")


if __name__ == "__main__":
    import sys
    if "--v2-only" in sys.argv:
        main_v2_only()
    else:
        main(stats_only="--stats" in sys.argv)
