"""
FastAPI SSE backend for the Taxxa demo frontend.

POST /ask streams phased events: plan, entry, hop, sources, token, done.
Reuses retrieve_with_graph + the OpenRouter synthesis call from answerer.py.
"""

import asyncio
import contextlib
import glob
import json
import os
import time
import urllib.parse
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import retriever
from answerer import ANSWER_MODEL, SYNTHESIS_SYSTEM, _get_client, _plan
from retriever import DATA_DIR, _load, _load_graph, format_nodes, retrieve, retrieve_with_graph


def _all_nodes() -> dict:
    """Resolve the retriever's live _nodes dict.

    `from retriever import _nodes` would capture the initial None value
    forever, because _load() rebinds the module attribute. Always go through
    the module to see the populated dict.
    """
    return retriever._nodes or {}


_parent_nodes: dict[str, dict] = {}


def _load_parent_nodes() -> None:
    """Cache parent_nodes.json so /corpus/stats is O(1)."""
    global _parent_nodes
    path = DATA_DIR / "parent_nodes.json"
    if not path.exists():
        _parent_nodes = {}
        return
    with path.open() as f:
        _parent_nodes = json.load(f) or {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm caches so the first /ask request doesn't pay 30+ seconds of cold load.
    print("Preloading retriever + graph...")
    _load()
    _load_graph()
    _load_parent_nodes()
    # Fail fast if the embedding model didn't actually populate the singleton —
    # otherwise the first /ask request silently pays 2-3s to load it.
    assert retriever._model is not None, "embedding model failed to load at startup"
    assert retriever._vectors is not None, "vectors failed to load at startup"
    assert retriever._bm25 is not None, "BM25 failed to load at startup"
    print(f"Ready. ({len(_parent_nodes)} parents indexed)")
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://getlaki.ayaanjamil.com",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    use_graph: bool = True
    hop_delay_ms: int = 300
    top_k: int = 10
    max_hops: int = 30  # cap traversal animation length so the demo stays watchable


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


_TAG_FROM_SOURCE = {
    "vero": ("vh", "guidance", "Guidance"),
    "finlex": ("finlex", "statute", "Statute"),
}


def _short_label(nid: str, node: dict) -> str:
    """Compact label for the Cytoscape node. Prefer statute+section, fall back to title head."""
    statute = node.get("statute")
    section = node.get("section")
    if statute and section:
        return f"§{section} {statute}"
    title = (node.get("title") or "").strip()
    if title:
        return title[:38] + ("…" if len(title) > 38 else "")
    return nid[-40:]


def _to_graph_node(node: dict) -> dict:
    """Map an internal chunk dict to the GraphNode shape the frontend expects."""
    parent_id = node.get("parent_id") or node["id"]
    ntype = (node.get("type") or "").upper() or "ARTICLE"
    if ntype not in {"ARTICLE", "CLAUSE", "GUIDANCE_S", "GUIDANCE", "COURT_CASE"}:
        ntype = "GUIDANCE" if node.get("source") == "vero" else "ARTICLE"
    text = (node.get("text") or "").strip().replace("\n", " ")
    return {
        "id": parent_id,
        "type": ntype,
        "label": _short_label(parent_id, node),
        "superseded": bool(node.get("superseded_by")),
        "desc": text[:140] + ("…" if len(text) > 140 else ""),
    }


def _chunk_label(chunk: dict) -> str:
    """Human-readable label for a single chunk citation chip tooltip."""
    statute = chunk.get("statute")
    section = chunk.get("section")
    if statute and section:
        return f"§{section} {statute}"
    src = chunk.get("source") or "doc"
    site = "vero.fi" if src == "vero" else "finlex.fi" if src == "finlex" else src
    if statute:
        return f"{statute} · {site} #chunk{chunk.get('chunk_index', '?')}"
    return f"{site} #chunk{chunk.get('chunk_index', '?')}"


def _source_url(node: dict) -> str | None:
    """Best-effort link out — full URLs aren't in the index, so we use the
    site's own search with the document title. Better than nothing for the demo;
    judges can verify a citation in one click."""
    title = (node.get("title") or "").strip()
    if not title:
        return None
    src = node.get("source")
    q = urllib.parse.quote_plus(title.replace(" - vero.fi", ""))
    if src == "vero":
        return f"https://www.vero.fi/search/?q={q}"
    if src == "finlex":
        return f"https://finlex.fi/fi/search/?text={q}"
    return None


def _to_source(gn: dict, node: dict, chunks_for_parent: list[dict]) -> dict:
    """Map a GraphNode + original chunk to the Source shape the chat panel renders.

    `chunks_for_parent` is every chunk in the answer's retrieval context that
    belongs to this parent — used both to resolve [chunk_id] citations and to
    show human-readable labels per chunk.
    """
    source = node.get("source") or "finlex"
    if gn["superseded"]:
        dot, tag, tag_label = "repealed", "repealed", "Repealed"
    elif gn["type"] == "COURT_CASE":
        dot, tag, tag_label = "court", "court", "Court"
    else:
        dot, tag, tag_label = _TAG_FROM_SOURCE.get(source, ("finlex", "statute", "Statute"))
    return {
        "id": gn["label"],
        "label": (node.get("title") or gn["id"])[:120],
        "dotType": dot,
        "tag": tag,
        "tagLabel": tag_label,
        "chunks": [
            {
                "id": c["id"],
                "label": _chunk_label(c),
                "score": c.get("_retrieval_score"),
                "rank": c.get("_retrieval_rank"),
            }
            for c in chunks_for_parent
        ],
        "parentId": gn["id"],
        "url": _source_url(node),
    }


def _dedupe_by_parent(chunks: list[dict]) -> list[dict]:
    """Keep first chunk per parent_id; order preserved."""
    seen: set[str] = set()
    out: list[dict] = []
    for c in chunks:
        pid = c.get("parent_id") or c["id"]
        if pid in seen:
            continue
        seen.add(pid)
        out.append(c)
    return out


async def _stream(req: AskRequest, request: Request) -> AsyncGenerator[str, None]:
    t0 = time.perf_counter()

    async def disconnected() -> bool:
        # Cheap to call repeatedly — checks a flag on the receive channel.
        try:
            return await request.is_disconnected()
        except Exception:
            return False

    # --- PLAN
    t_plan_start = time.perf_counter()
    sub_questions = _plan(req.question)
    plan_ms = int((time.perf_counter() - t_plan_start) * 1000)
    yield sse("plan", {"sub_questions": sub_questions, "elapsed_ms": plan_ms})

    # --- RETRIEVE (entry chunks + graph traversal log)
    # Per-sub-question entry events let the frontend color graph nodes by
    # which sub-question retrieved them — directly visualizes the planner's
    # decomposition. Graph-walk endpoints (added below) get sub_idx=null.
    t_retrieve_start = time.perf_counter()
    all_chunks: list[dict] = []
    # Track which sub-q first retrieved each parent_id (lowest index wins).
    parent_sub_idx: dict[str, int] = {}
    # Track each chunk's source sub-q for downstream score / score-table needs.
    traversal_log: list[dict] = []
    queries = sub_questions or [req.question]
    for idx, sq in enumerate(queries):
        if await disconnected():
            return
        t_sq = time.perf_counter()
        if req.use_graph:
            chunks, hops = retrieve_with_graph(sq, top_k=req.top_k)
            traversal_log.extend(hops)
        else:
            chunks = retrieve(sq, top_k=req.top_k)
        all_chunks.extend(chunks)

        # Tag entry nodes with sub_idx. Dedupe by parent within this sub-q so
        # the frontend doesn't get the same node twice in one batch. A parent
        # already claimed by an earlier sub-q stays with the earlier sub-q.
        sq_nodes: list[dict] = []
        seen_in_sq: set[str] = set()
        for c in chunks:
            pid = c.get("parent_id") or c["id"]
            if pid in seen_in_sq or pid in parent_sub_idx:
                continue
            seen_in_sq.add(pid)
            parent_sub_idx[pid] = idx
            gn = _to_graph_node(c)
            gn["subIdx"] = idx
            sq_nodes.append(gn)
        if sq_nodes:
            yield sse("entry", {"nodes": sq_nodes, "sub_idx": idx})
        yield sse("sub_done", {"index": idx, "hits": len(chunks), "elapsed_ms": int((time.perf_counter() - t_sq) * 1000)})
    retrieve_ms = int((time.perf_counter() - t_retrieve_start) * 1000)

    # --- HOPS (paced) — only emit hops whose endpoints we can shape into graph nodes
    # Cap the traversal so the demo stays watchable (--reload runs can have 200+ hops).
    t_hops_start = time.perf_counter()
    capped_log = traversal_log[: req.max_hops]
    known_parents = set(parent_sub_idx.keys())
    for h in capped_log:
        if await disconnected():
            return
        # Ensure both endpoints exist in the frontend graph before drawing the edge.
        for endpoint in (h["from"], h["to"]):
            if endpoint in known_parents:
                continue
            # Look up any chunk for this parent so we can build a GraphNode for it.
            chunk = next(
                (c for c in _all_nodes().values() if (c.get("parent_id") or c["id"]) == endpoint),
                None,
            )
            if chunk is not None:
                node_payload = _to_graph_node(chunk)
                # Graph-walk endpoints aren't from any sub-question — null sub_idx
                # so the frontend renders them with a neutral border.
                node_payload["subIdx"] = None
                known_parents.add(endpoint)
                yield sse("entry", {"nodes": [node_payload], "sub_idx": None})

        yield sse("hop", {"from": h["from"], "to": h["to"], "relation": h["relation"]})
        await asyncio.sleep(req.hop_delay_ms / 1000)
    hops_ms = int((time.perf_counter() - t_hops_start) * 1000)
    deduped = _dedupe_by_parent(all_chunks)

    # --- SOURCES
    # Bucket every chunk in the retrieval context by its parent so the frontend
    # can resolve [chunk_id] citations back to a source row AND show a human
    # label for each chunk.
    parent_to_chunks: dict[str, list[dict]] = {}
    seen_chunk_ids: set[str] = set()
    for c in all_chunks:
        if c["id"] in seen_chunk_ids:
            continue
        seen_chunk_ids.add(c["id"])
        pid = c.get("parent_id") or c["id"]
        parent_to_chunks.setdefault(pid, []).append(c)
    sources_chunks = deduped[:25]
    sources = [
        _to_source(
            _to_graph_node(c),
            c,
            parent_to_chunks.get(c.get("parent_id") or c["id"], [c]),
        )
        for c in sources_chunks
    ]
    yield sse(
        "sources",
        {
            "sources": sources,
            "hops": len(capped_log),
            "nodes": len(known_parents),
            "time_ms": int((time.perf_counter() - t0) * 1000),
        },
    )

    # --- SYNTHESIS (stream tokens)
    t_synth_start = time.perf_counter()
    context = format_nodes(all_chunks, max_chars=14000)  # unified with answerer.py
    user_prompt = (
        f"Question: {req.question}\n\n"
        f"Sub-questions researched: {sub_questions}\n\n"
        f"Documents:\n{context}"
    )
    stream = _get_client().chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=800,
        temperature=0.1,
        stream=True,
    )
    try:
        for chunk in stream:
            # If the client aborted, stop iterating AND close the upstream
            # stream so OpenRouter stops generating (we keep getting billed
            # otherwise — synthesis can run for 20+ seconds after disconnect).
            if await disconnected():
                return
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield sse("token", {"text": delta})
            # Yield control so the event loop can flush the chunk to the client.
            await asyncio.sleep(0)
    except asyncio.CancelledError:
        # FastAPI cancels the generator task on client disconnect — let it propagate
        # after finally closes the upstream LLM stream.
        raise
    finally:
        with contextlib.suppress(Exception):
            stream.close()

    synth_ms = int((time.perf_counter() - t_synth_start) * 1000)
    yield sse(
        "done",
        {
            "time_ms": int((time.perf_counter() - t0) * 1000),
            "phase_ms": {
                "plan": plan_ms,
                "retrieve": retrieve_ms,
                "hops": hops_ms,
                "synth": synth_ms,
            },
        },
    )


@app.post("/ask")
async def ask(req: AskRequest, request: Request) -> StreamingResponse:
    return StreamingResponse(_stream(req, request), media_type="text/event-stream")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model": ANSWER_MODEL}


@app.get("/chunk")
def get_chunk(id: str) -> dict:
    """Return the full chunk text for a citation drawer."""
    c = _all_nodes().get(id)
    if c is None:
        raise HTTPException(status_code=404, detail="chunk not found")
    return {
        "id": c["id"],
        "parentId": c.get("parent_id"),
        "title": c.get("title"),
        "statute": c.get("statute"),
        "section": c.get("section"),
        "date": c.get("date"),
        "source": c.get("source"),
        "type": c.get("type"),
        "chunkIndex": c.get("chunk_index"),
        "chunkTotal": c.get("chunk_total"),
        "text": c.get("text") or "",
        "filePath": c.get("file_path"),
        "label": _chunk_label(c),
        "url": _source_url(c),
    }


_TIER_BUCKETS = {
    "basic":  ("basic", "difficulty_1", "difficulty_2"),
    "medium": ("medium", "difficulty_3"),
    "hard":   ("hard", "difficulty_4", "difficulty_5"),
}


def _bucket_color(pct: float) -> str:
    if pct >= 55: return "green"
    if pct >= 30: return "amber"
    return "red"


@app.get("/eval/latest")
def eval_latest() -> dict:
    """Read the newest data/eval_runs/*_graph.json and aggregate by tier."""
    files = sorted(glob.glob(str(DATA_DIR / "eval_runs" / "*_graph.json")))
    if not files:
        return {"available": False}
    latest = files[-1]
    with open(latest) as f:
        data = json.load(f)
    by_tier_scores: dict[str, list[float]] = {}
    by_tier_cites: dict[str, list[bool]] = {}
    for r in data.get("results", []):
        t = r.get("tier") or "unknown"
        by_tier_scores.setdefault(t, []).append(float(r.get("fact_score", 0.0)))
        by_tier_cites.setdefault(t, []).append(bool(r.get("has_citation")))
    buckets = []
    for label, tiers in _TIER_BUCKETS.items():
        scores = [s for t in tiers for s in by_tier_scores.get(t, [])]
        if not scores:
            continue
        pct = round(100 * sum(scores) / len(scores))
        buckets.append({"label": label.capitalize(), "pct": pct, "n": len(scores), "color": _bucket_color(pct)})
    return {
        "available": True,
        "filename": Path(latest).name,
        "mtime": os.path.getmtime(latest),
        "total": len(data.get("results", [])),
        "overallFactCoverage": data.get("overall_fact_coverage"),
        "overallCitationRate": data.get("overall_citation_rate"),
        "buckets": buckets,
    }


@app.get("/corpus/stats")
def corpus_stats() -> dict:
    """Counts derived from parent_nodes.json. Loaded once at startup."""
    if not _parent_nodes:
        return {"available": False}
    finlex_statutes = Counter()
    type_by_source: dict[str, Counter] = {"finlex": Counter(), "vero": Counter()}
    for p in _parent_nodes.values():
        src = p.get("source")
        ptype = p.get("type") or "UNKNOWN"
        if src in type_by_source:
            type_by_source[src][ptype] += 1
        if src == "finlex" and p.get("statute"):
            finlex_statutes[p["statute"]] += 1
    vero_total = sum(type_by_source["vero"].values())
    court_total = type_by_source["finlex"]["COURT_CASE"]
    return {
        "available": True,
        "totalParents": len(_parent_nodes),
        "statutes": {
            "count": len(finlex_statutes),
            "items": [{"name": s, "detail": f"{n} parents"} for s, n in finlex_statutes.most_common()],
        },
        "vero":  {"count": vero_total, "items": []},
        "court": {"count": court_total, "items": []},
    }
