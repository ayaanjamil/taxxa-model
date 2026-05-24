# Taxxa — Architecture & Lookup Walkthrough

A GraphRAG system for Finnish tax law. Answers natural-language questions
(Finnish or English) over Finlex statutes + Verohallinto guidance with cited
sources, and visualizes the retrieval path as a graph.

This document explains **how the system is built** and **what happens during a
single `/ask` call**.

---

## 1. High-Level Architecture

```
                ┌───────────────────────────────────────────────────────────┐
                │   finland_kb/   (63k Finnish HTML files: Finlex + Vero)   │
                └──────────────────────────┬────────────────────────────────┘
                                           │ parser.py (one-shot, offline)
                                           ▼
            ┌────────────────────────────────────────────────────────┐
            │   data/nodes.json       — 376,200 chunks (~800 chars)  │
            │   data/parent_nodes.json — 37,212 parent documents     │
            └─────────────────┬──────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┬───────────────────────┐
              ▼               ▼               ▼                       ▼
      embedder.py /     retriever.py    graph_builder.py        eval_harness.py
      modal_embed.py    (BM25 index)    extract_edges.py        (83 Q&A pairs,
       (one-shot)                       (deterministic+         LLM-as-judge)
              │                          LLM-typed edges)
              ▼               ▼               ▼
      vectors.npy        bm25.pkl         graph.pkl
      id_map.json        voikko_cache     graph_v2.pkl
       (or _bgem3)

                              │
                              ▼
         ┌──────────────────────────────────────────────────────┐
         │  retriever.py  — hybrid retrieval (vector + BM25 +   │
         │                  §-expansion + graph walk)           │
         └─────────────────┬────────────────────────────────────┘
                           │
                           ▼
         ┌──────────────────────────────────────────────────────┐
         │  answerer.py   — planner → retrieve → RRF → synth    │
         └─────────────────┬────────────────────────────────────┘
                           │
                           ▼
         ┌──────────────────────────────────────────────────────┐
         │  api.py        — FastAPI SSE: plan / entry / hop /   │
         │                  sources / token / done events       │
         └─────────────────┬────────────────────────────────────┘
                           │
                           ▼
                   Next.js + Cytoscape.js frontend
                   (graph viz + streamed answer)
```

Every box is one Python file. There is no database, no message queue, no
container orchestrator — just files on disk and a FastAPI process.

---

## 2. Files and Responsibilities

| File | Role | When it runs |
|---|---|---|
| `parser.py` | Walks `finland_kb/`, extracts text from Finnish HTML, splits into ~800-char chunks, preserves statute/section metadata, extracts `§NN` references. Writes `data/nodes.json` and `data/parent_nodes.json`. | Once, offline. |
| `embedder.py` / `modal_embed.py` | Embeds every chunk with a multilingual sentence-transformer. Local CPU path (mpnet) or Modal A10G path (BGE-M3). Writes `data/vectors*.npy` + `data/id_map*.json`. | Once per embedder swap. |
| `graph_builder.py` | Builds a parent-level NetworkX DiGraph. Deterministic `cites` edges (Vero → Finlex via §) and `amends` edges (Finlex amendment headers). Writes `data/graph.pkl`. | Once, offline. |
| `extract_edges.py` | (Optional) Re-types deterministic edges into semantic types (`interpreted_by`, `clarified_by`, `overrides`) via an LLM. Writes `data/graph_v2.pkl`. | Once, opt-in. |
| `retriever.py` | Lazy-singleton loader for model + vectors + BM25 + graph. Exposes `retrieve()` and `retrieve_with_graph()`. | Live, every query. |
| `answerer.py` | The brain. Planner LLM → parallel retrievals → RRF merge → synthesis LLM (+ Perplexity web fallback on refusal). | Live, every query. |
| `api.py` | FastAPI app. `/ask` is an SSE endpoint that streams the answer + the graph that produced it. | Live, every query. |
| `eval_harness.py` | Runs the 83-question QA bank, scores fact coverage with an LLM judge, prints per-tier results. | On demand. |

---

## 3. Data Model

### Chunk (`nodes.json`)
```jsonc
{
  "id": "vero_Apteekkien_..._Apteekkien_ennakkomaksut...#chunk0",
  "parent_id": "vero_Apteekkien_..._Apteekkien_ennakkomaksut...",
  "chunk_index": 0,
  "chunk_total": 6,
  "source": "vero" | "finlex",
  "type":   "GUIDANCE" | "GUIDANCE_S" | "ARTICLE" | "CLAUSE" | "STATUTE" | "COURT_CASE",
  "statute": "TVL" | "AVL" | "EPL" | null,
  "title": "Apteekkien ennakkomaksut nettovarallisuudessa - vero.fi",
  "text":  "Asian kuvaus Verotuskäytännössä on havaittu...",
  "date":  "2024-01-01" | null,
  "superseded_by": null,
  "references": ["TVL_§38", "TVL_§30"],
  "file_path": "finland_kb/vero/.../Apteekkien....html"
}
```

### Parent-level graph (`graph.pkl`)
- **Nodes**: 37,212 parent documents (one per HTML file).
- **Edges**: 1,706 total — 1,048 `cites` (Vero → Finlex section), 658 `amends`
  (Finlex amendment → base law).
- All retrieval indices are aligned to `id_map.json` ordering (vectors, BM25,
  node list) so an index `i` always refers to the same chunk.

---

## 4. Indices Built Once, Loaded Lazy

`retriever._load()` runs at FastAPI startup (`lifespan`) so the first request
doesn't pay cold-start tax. It loads:

| Singleton | Source file | Memory | Notes |
|---|---|---|---|
| `_model` | sentence-transformers from HF | ~500 MB | Encodes the query at request time. Uses MPS on Apple Silicon, CUDA on GPU boxes, CPU elsewhere. |
| `_vectors` | `data/vectors*.npy` | 1.1–1.5 GB | `np.load`, kept resident. Cosine sim via dot product (vectors are L2-normalized). |
| `_id_map` | `data/id_map*.json` | 37 MB | `int → chunk_id` and the inverse. |
| `_nodes` | `data/nodes.json` | 512 MB | Full chunk dicts, keyed by chunk_id. |
| `_node_list` | derived | — | Parallel-to-`_vectors` ordering, used for BM25 index. |
| `_bm25` | `data/bm25.pkl` | 448 MB | Pre-tokenized `BM25Okapi`. Tokens are Voikko base forms (Finnish morphology). |
| `_section_index` | derived | small | `"TVL_§102" → [chunk_ids]` for §-ref expansion. |
| `_graph` | `data/graph.pkl` | small | Parent-level DiGraph, lazy-loaded only when graph retrieval is used. |
| `_voikko_cache` | `data/voikko_cache.json` | 20 MB | Precomputed `token → baseform` to skip libvoikko at query time. |

**Why two indices (vector + BM25)?** Embeddings smear pension docs together
(TyEL ≈ VEL ≈ lisäeläkevakuutus) — semantic similarity blurs the exact statute
you need. BM25 over Voikko-lemmatized text catches *exact* matches on Finnish
acronyms and inflected forms ("TyEL", "syntymävuosi") that the embedder
collapses. RRF merges both ranked lists.

---

## 5. What Happens During a Single `/ask` Call

A user POSTs `{ "question": "What withholding-tax rate applies to a foreign
specialist with key-personnel status?" }` to `/ask`. Here's the exact sequence
(file:line references for navigation).

### Phase 1 — PLAN (`answerer._plan`, ~1–2 s)

The planner LLM (`anthropic/claude-haiku-4-5` by default) decomposes the user
question into 2–4 Finnish keyword strings. The system prompt
(`answerer.py:64-126`) is heavily tuned: it forces use of Finnish base stems
(`pääomatulo`, not `pääomatulovero`), expands legal acronyms (TyEL, AVL), and
contains explicit rules for treaty questions (every sub-q must include the
country name AND article keywords in the same string, because treaty article
bodies don't repeat the country name).

```python
# answerer.py:141-164
sub_questions = _plan(question)
# → ["avainhenkilö lähdevero prosentti",
#    "ulkomailta tuleva palkansaaja avainhenkilölaki verokortti voimassaolo"]
```

Emits SSE event: `plan { sub_questions, elapsed_ms }`.

### Phase 2 — RETRIEVE (per sub-question, ~200–500 ms each)

For each sub-question, the retriever runs `retrieve_with_graph(sq, top_k=10)`
(or `retrieve()` if graph mode is off). Internally:

1. **Acronym expansion** — `expand_query()` replaces standalone `TyEL` with
   `Työntekijän eläkelaki (TyEL)` so the embedder has real semantic surface.
2. **Vector search** — encode the expanded query, dot-product against
   `_vectors`, take top `candidate_pool=50`.
3. **BM25 search** — tokenize the query through `_tokenize_fi` (regex tokens
   + Voikko base forms), score against the cached BM25 index, take top 50.
4. **RRF merge** — `score(idx) = Σ 1/(60 + rank_in_list)` across both lists.
   Reciprocal Rank Fusion is rank-only, so vector and BM25 don't need score
   normalization.
5. **Authority boosts** —
   - `vero` source × 1.15 (Verohallinto guidance is the most current,
     consolidated source).
   - Bilateral treaty docs (`"Tuloverosopimukset" in id`) × 1.15.
6. **§-reference expansion** — for each top-ranked chunk, follow its
   `references` list (e.g. `TVL_§102`) through `_section_index` and add up to
   2 referenced chunks per ref at `0.6 × parent_score`.
7. **Treaty neighbor expansion** — bilateral treaty chunks have no §-refs, so
   pull in ±3 adjacent chunks of the same document at `0.75 × parent_score`
   (Article 10 dividend rates sit a few chunks past the title-only preamble).
8. **Re-sort** — non-superseded first, then by score.
9. **Parent cap** — keep at most 2 chunks per parent doc (4 for treaty docs)
   so `top_k` surfaces distinct sources, not five fragments of the same page.
10. **Graph walk** (if `use_graph=True`) — the top chunks' parent IDs become
    "entry parents." For up to `max_hops=2` hops, follow `cites`, `amends`,
    `interpreted_by`, `overrides` successors and `amends` predecessors. Each
    discovered parent contributes its own chunks, ranked by type priority
    (statute > guidance > court ruling) and date.

Per sub-question, the API streams:
- `entry { nodes, sub_idx }` — graph nodes color-coded by which sub-q first
  retrieved them.
- `sub_done { index, hits, elapsed_ms }`.

After all sub-questions, the API paces and streams `hop { from, to, relation }`
events at `hop_delay_ms=300` so the user *sees* the traversal animate.

### Phase 3 — MERGE (`answerer._merge_nodes`)

All sub-question retrievals are merged again with RRF (different from the
intra-retrieval RRF: this one operates over node lists, not rank dicts). A
node appearing high across multiple sub-questions wins. Output: a single
deduped list ordered by `Σ 1/(60 + rank)`.

### Phase 4 — SYNTHESIS (`anthropic/claude-haiku-4-5`, ~3–6 s streamed)

The merged chunks are formatted into a context block capped at 18,000
characters (`TAXXA_CTX_CHARS`, was 14k pre-tuning). Each chunk renders as:

```
[node_id] Title (source, type) — date
<chunk text>

---
```

The synthesis system prompt (`answerer.py:129-138`) constrains:
- Every factual claim must cite `[node_id]`.
- On conflicting dates, prefer the newest (supersession).
- Prefer Verohallinto guidance over Finlex amendment history when both cover
  the same fact.
- Answer in the same language as the question.

The LLM call is streamed; each delta token is forwarded to the client as
`token { text }`. If the client disconnects mid-stream, the upstream OpenAI
stream is closed so OpenRouter stops billing for tokens nobody is reading.

### Phase 5 — REFUSAL → WEB FALLBACK (eval mode skips this)

If the answer contains any of ~20 refusal markers (Finnish + English: "en
löydä", "cannot find", "ei sisällä", …), `answerer._web_fallback` calls
Perplexity Sonar with a Finnish-tax-authority system prompt. This is what
keeps live-demo answers from looking like dead-ends; for eval runs it's off so
we measure pure RAG performance.

### Phase 6 — SOURCES + DONE

After synthesis, the API emits:
- `sources { sources[], hops, nodes, time_ms }` — each source has a label,
  tag (Statute / Guidance / Court / Repealed), per-chunk citation chips, and
  a "best-effort" outbound link (vero.fi / finlex.fi search by title).
- `done { time_ms, phase_ms: { plan, retrieve, hops, synth } }` — end of stream.

---

## 6. The Frontend's Job

A Next.js page subscribes to the SSE stream and:
- Renders the **graph** in Cytoscape.js, color-coded by node type and by which
  sub-question retrieved it. Superseded nodes are gray. Edges are labeled by
  relation type.
- Animates the **traversal path** in yellow as `hop` events arrive, paced by
  `hop_delay_ms` so the user can follow it.
- Streams the **answer text** into the right panel as `token` events arrive.
- After `sources` arrives, resolves `[chunk_id]` citations in the answer text
  to clickable chips that open a drawer with the full chunk text (served by
  `GET /chunk?id=…`).

A dashboard widget reads `GET /eval/latest` to show the most recent eval run's
per-tier scores next to the live demo — judges can see the system's measured
performance while testing it.

---

## 7. Eval Pipeline

`eval_harness.py` loads `question_bank.json` (83 graded QA pairs, tiered by
difficulty), calls `answer(q, allow_web_fallback=False)` per question, and for
each `answer_key_fact` calls an LLM judge:

```
System: Strict fact-checker. Return JSON {covered: true/false, reason: ...}.
        The fact and answer may be in Finnish or English. Evaluate
        semantically, not lexically. A paraphrase counts; a contradiction
        does not.
User:   Fact to check: "..."
        Answer to check: "..."
```

Per-question score = covered_facts / total_facts. Per-tier averages are
printed, results saved as JSON to `data/eval_runs/`.

Flags:
- `--ids Q1,N7,…` run only specific questions (for stable A/B comparison).
- `--sample N --seed S` random subset for fast iteration.
- `--tier hard` filter by difficulty bucket.
- `--mode graph` use graph-walk retrieval instead of plain RRF.
- `--dry-run` skip the judge calls.

---

## 8. Tunable Knobs (env vars)

| Var | Default | What it does |
|---|---|---|
| `TAXXA_EMBED_MODEL` | `mpnet` | `mpnet` (768-dim, legacy) or `bge-m3` (1024-dim, Modal-embedded). |
| `TAXXA_VECTORS_FILE` | derived | Override `vectors*.npy` path for ablations. |
| `TAXXA_ID_MAP_FILE` | derived | Override `id_map*.json` path. |
| `TAXXA_GRAPH_VERSION` | `v1` | `v1` deterministic edges only; `v2` adds LLM-typed edges (regressed in testing — keep v1). |
| `TAXXA_CTX_CHARS` | `18000` | Synthesis context cap. 14k drops late evidence; 20k bleeds noise on Haiku. |
| `TAXXA_DEDUP_MODE` | `post` | Dedup chunks after RRF (`post`) or collapse to unique parents before RRF (`pre`). |
| `TAXXA_RRF_K` | `60` | RRF smoothing constant. 60 = stronger top-rank emphasis; 100 = softer. |
| `TAXXA_ENTRY_FRONTIER_MULT` | `3` | Multiplier for entry chunk count in graph walk. Wider frontier → catches more hard-tier hits. |
| `TAXXA_ANSWER_MODEL` | `anthropic/claude-haiku-4-5` | Synthesis LLM. |
| `TAXXA_PLANNER_MODEL` | `anthropic/claude-haiku-4-5` | Planner LLM. |
| `TAXXA_JUDGE_MODEL` | `anthropic/claude-haiku-4-5` | Eval-judge LLM. |
| `TAXXA_WEB_MODEL` | `perplexity/sonar` | Web fallback on corpus refusal. |
| `OPENROUTER_API_KEY` | — | Required for any LLM call. |

---

## 9. Why this design, in one paragraph

The corpus is Finnish, the questions are multilingual, the answer terms span
multiple documents, and amendments mean *recency matters*. Naive vector RAG
fails on three of those four. The fix is layered: BM25 over Voikko-lemmatized
text restores exact-match precision a multilingual embedder destroys; explicit
§-reference and graph traversal restore the cross-document links that
embeddings can't see; the planner restores multi-fact decomposition that
single-query RAG can't express; supersession metadata + vero-authority boosts
restore recency. The synthesis step is intentionally dumb — it just composes
cited claims from evidence the retrievers already curated.
