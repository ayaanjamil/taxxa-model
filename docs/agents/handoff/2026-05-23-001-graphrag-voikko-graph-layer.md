# Handoff: Voikko Tokenizer, Treaty Retrieval Fixes, and Graph Layer

**Date:** 2026-05-23
**Branch:** `master` (no commits made this session — all changes uncommitted)
**Plan file:** `/Users/ayaan/.claude/plans/look-at-what-we-delegated-book.md`
**Project context:** Taxxa hackathon — Finnish tax law GraphRAG over Finlex + Verohallinto corpus (376k chunks / 37k parent docs, 83 graded QA pairs)

---

## What was accomplished

### 1. Retrieval fixes (driven by Q20 "Finland-Austria treaty" failure)

The canonical failure case was **Q20** in `question_bank.json`: a multi-hop bilateral-treaty question about dividend withholding. Before fixes, the system returned 0% fact coverage and never surfaced the treaty document — even though `finland_kb/finlex/Tuloverosopimukset/Itävalta/` exists and `nodes.json` contains 266 chunks from it, including chunk39 with the exact Article 10 dividend rate text.

Root causes identified:
- **BM25 morphology mismatch**: planner emitted `"Itävalta"` (nominative); document title contained `"Itävallan"` (genitive). BM25 does exact token matching — no overlap.
- **Vero authority boost (1.15×) buried treaties**: `Tuloverosopimukset` nodes are `source: finlex` so they got no boost, while every Vero guidance doc did.
- **Article body lacks the country name**: the treaty title says "Itävalta" but chunk39's text only says "sopimusvaltio" (contracting state). A query matching only the title won't surface the article body.
- **Parent cap of 2 chunks/parent** kept chunk39 out: chunk0/chunk6/chunk16 (preamble) ranked higher than chunk39 and consumed both slots.

Fixes applied to `retriever.py`:
- **Voikko Finnish morphology**: `libvoikko` (Homebrew + pip) integrated into `_tokenize_fi`. Each token now emits both the original form and its Voikko BASEFORM (lowercased). E.g., `itävallan` → `["itävallan", "itävalta"]`. Verified Voikko speed: ~44k calls/sec, ~7s for 300k unique tokens — negligible vs BM25 build cost.
- **Treaty boost**: nodes whose ID contains `"Tuloverosopimukset"` get the same 1.15× boost as `vero` nodes.
- **Treaty neighbor expansion** (±3 chunks): when any `Tuloverosopimukset` chunk matches, adjacent chunks get pulled into the candidate pool with score × 0.75. Surfaces article body when only preamble matched.
- **Treaty parent cap = 4** (was 2 for everyone): long structured treaty docs can contribute up to 4 chunks so multiple articles (dividend / interest / PE) co-exist in the synthesis context.

Fix applied to `answerer.py` (`PLANNER_SYSTEM`):
- Treaty-specific instruction: **every sub-query MUST combine country name + article-content keywords in the same string**. Generic article-only queries match Article 10 of every bilateral treaty (Singapore, Kazakhstan, etc.) and dilute the signal. Example added: `"Itävalta osinko äänimäärä hallitsee 10 prosenttia lähdevero"`.

Result: chunk39 now ranks **#1** for the article-content sub-queries. Q20 baseline answer is correct (0% rate, cites chunk39, applies Article 10(3)).

### 2. Persistent caches (cold-start optimization)

`retriever.py` now writes and reuses:
- `data/bm25.pkl` (446 MB) — pickled `BM25Okapi` object
- `data/voikko_cache.json` (19 MB) — `dict[str, str]` mapping token → baseform

Cold-load time: **78.6s → 32.4s** (first run builds and saves; subsequent runs load).
Remaining 32s is dominated by `vectors.npy` (1.1 GB) + `nodes.json` (507 MB) + `bm25.pkl` (446 MB) deserialization.

### 3. Graph layer (`graph_builder.py` — new file)

Builds a typed networkx `DiGraph` at **parent document level** (37,212 parents — not 376k chunks; chunk-level would use ~450 MB RAM).

Three deterministic passes — **no LLM required**:
- **Pass 1** `build_parent_nodes()`: collapse chunks into parents; merge `references` lists; keep first non-null date.
- **Pass 2** `build_section_index()`: build `{STATUTE_§N → [parent_id, ...]}` from both the `references` field AND from §-numbers extracted from amendment titles. Handles the Finnish `"69 ja 71 §"` pattern (`_SEC_JA_RE`) that the bare `_SEC_RE` misses.
- **Pass 3** `build_edges()`:
  - **`cites`**: vero parent → finlex parent(s) that define a cited §. 1,048 edges.
  - **`amends`**: finlex amendment parent → finlex base-law parent(s) for the amended §. Detected via `"muuttamisesta"` in title. 658 edges.

Outputs:
- `data/graph.pkl` (7.6 MB) — pickled `DiGraph` (networkx 3.x dropped `write_gpickle`; we use `pickle.dump` directly)
- `data/graph_edges.json` (472 KB) — edge list for debugging
- `data/parent_nodes.json` (13 MB) — parent registry

Run: `python graph_builder.py` (full build) or `python graph_builder.py --stats` (dry run, no save).

### 4. Graph-walk retrieval (`retrieve_with_graph()` in `retriever.py`)

Additive — `retrieve()` is unchanged.

Signature: `retrieve_with_graph(query, top_k=10, candidate_pool=50, max_hops=2, hop_relations=("cites", "amends", "interpreted_by", "overrides"), frontier_cap=20) -> (chunks, traversal_log)`

Flow:
1. Call `retrieve()` with `top_k * 2` to get entry chunks with headroom.
2. Group entry chunks by `parent_id`; record which specific chunks BM25/vector matched (`entry_parent_chunks`).
3. Walk outgoing edges of allowed types + incoming `amends` edges (to surface newer versions). Frontier capped at 20 per hop.
4. **Critical**: when re-expanding parents → chunks, use the BM25-ranked entry chunks for entry parents (not natural chunk order). Falls back to natural order only for graph-expanded parents. This was the bug discovered live — without it, graph mode lost the specific chunk match (e.g., chunk39) and only returned chunk0..chunk3 of the treaty preamble.
5. Apply the same parent cap as `retrieve()` (4 for treaties, 2 elsewhere).

`traversal_log` shape (for future frontend / API): `[{"from": parent_id, "to": parent_id, "relation": str}, ...]`

### 5. Eval harness & answerer changes

`answerer.py`:
- `_local_answer` now returns `(answer_text, traversal_log)` and accepts `use_graph: bool`.
- `answer()` accepts `use_graph` parameter and unpacks the tuple internally.

`eval_harness.py`:
- New `graph_answer()` function (mirrors `real_answer()` but with `use_graph=True`).
- `real_answer()` now passes `allow_web_fallback=False` for fair RAG-only comparison.
- New `--mode {baseline,graph}` CLI flag.

A/B test command pattern (used live):
```bash
set -a && source .env && set +a && .venv/bin/python3 -u -c "
from answerer import answer
q = '...'
print('=== BASELINE ==='); print(answer(q, allow_web_fallback=False, use_graph=False))
print('=== GRAPH ==='); print(answer(q, allow_web_fallback=False, use_graph=True))
"
```

---

## Key decisions

1. **Voikko over simple suffix stripping or character n-grams.** Simple `-n` stripping fails on Finnish consonant gradation (`itävalta` → `itävallan` has `t` → `ll` stem change). Voikko gives true morphological base forms and runs fast enough (~44k tokens/sec). Trade-off: adds `libvoikko` Homebrew dep and 19 MB token cache.

2. **Graph at parent level, not chunk level.** 37k parent nodes vs 376k chunk nodes. Memory budget alone forces this. Consequence: graph node IDs are parent IDs; `_graph.successors(pid)` returns parent IDs; we maintain a separate `_parent_to_chunks` map to expand back to retrievable chunks.

3. **Preserve BM25-ranked chunks for entry parents in graph mode.** Discovered as a regression during live A/B: collapsing entry chunks to parents and re-expanding via natural chunk order lost the specific match (chunk39 → returned chunk0..chunk3). Fix uses `entry_parent_chunks` dict to remember which chunks BM25 actually picked, and falls back to natural order only for graph-expanded parents reached via hops.

4. **No LLM relation extraction (yet).** Plan called for `interpreted_by`/`overrides`/typed-edge classification via Claude. Skipped because:
   - Regex-only build gives 1,706 edges of high confidence (cites + amends) — enough to demo graph value.
   - Court nodes have empty `references` (regex didn't fire on them in the parser), so `overrides` edges would require LLM. Punt to later.
   - LLM classification cost on 500+ pairs is significant and time-boxed in the plan.

5. **Pickle `BM25Okapi` directly.** `rank_bm25` doesn't provide a serialization API; the object is plain Python data so `pickle.HIGHEST_PROTOCOL` works. Resulting 446 MB file loads in ~5s vs 30s rebuild. `nx.write_gpickle` was removed in networkx 3.x — use `pickle.dump` for the graph too.

6. **Single combined country+article queries in the planner.** Initial fix was two separate sub-queries (one targeting title, one targeting article content). This failed because the article-content query matched Article 10 of *every* bilateral treaty. Replaced with strict instruction: every treaty sub-query must contain both country name AND article keywords.

7. **Frontend recommendation: upgrade existing `frontend_mockup.html`, not Next.js.** The mockup already has the complete Cytoscape.js integration. Migration is a single `fetch()` swap. Documented in the plan file but not executed this session.

---

## Important context for future sessions

### File locations
- All data files in `/Users/ayaan/Projects/taxxa/data/` — gitignored via `/data` rule in `.gitignore`
- Corpus: `/Users/ayaan/Projects/taxxa/finland_kb/` (584 MB, also gitignored)
- Plan file: `/Users/ayaan/.claude/plans/look-at-what-we-delegated-book.md`
- Eval question bank: `question_bank.json` (83 entries, tracked in git)

### Data scale (use these numbers, not the stale ones in plan.md)
- 376,200 chunks (`nodes.json` 507 MB)
- 37,212 unique parent documents
- 376,200 vectors @ 768-dim float32 (`vectors.npy` 1.1 GB)
- Graph: 37,212 nodes, 1,171 unique edges (1,706 raw — networkx DiGraph deduplicates parallel)
- 954 unique `STATUTE_§N` references in the section index
- 200+ bilateral tax treaties under `finland_kb/finlex/Tuloverosopimukset/<country>/`

### Cache regeneration
If `nodes.json`/`vectors.npy` are re-built:
- Delete `data/bm25.pkl` and `data/voikko_cache.json` — they will rebuild on next retriever load.
- Re-run `python graph_builder.py` to rebuild `data/graph.pkl`, `graph_edges.json`, `parent_nodes.json`.

### Known issues / not-yet-fixed
- **LLM threshold reasoning**: occasionally the synthesis model misjudges threshold conditions (e.g., Q20 second company at 20% — the model wrote "Does not qualify (20% > 10% threshold)", which is self-contradictory). This is a model error, not a retrieval error. Not fixed.
- **N57 (autovero/van VAT) flakiness**: scored 33% in one user-run baseline, then 0% on a later 3-sample run. Small sample, low signal — needs a real eval run to characterize.
- **Graph mode at parity with baseline on single-hop questions**, as expected. Graph advantage should show on multi-hop hard-tier questions where the BM25/vector search alone misses connections. Not yet measured with full eval.
- **No real BM25 cache invalidation**: if `_tokenize_fi` logic changes, you must manually `rm data/bm25.pkl` to force a rebuild.

### What the plan says is next (and is NOT done)
From `/Users/ayaan/.claude/plans/look-at-what-we-delegated-book.md`:
- **`api.py`** — FastAPI backend with `POST /ask` returning `{answer, graph: {nodes, edges}, sources, elapsed_ms, hops}`. Plan has detailed pseudocode including the `_answer_with_graph` bridge function and startup preload to avoid first-request timeout.
- **Frontend hookup** — replace the `setTimeout` mock in `frontend_mockup.html`'s `runQuery()` with a real `fetch('http://localhost:8000/ask', ...)`. The mockup's `SAMPLE_GRAPH` format already matches the planned API response shape.
- **Verifier in `answerer.py`** — `_verify(answer, context)` gated on `TAXXA_VERIFY` env var. Plan has full implementation sketch.
- **Iterative search** (user-suggested, not in plan): production Taxxa does progressive search ("Finland Austria treaty" → refined with SopS code → refined with Finnish technical terms). Would replace the one-shot planner with a search→evaluate→refine agent loop.
- **Full 83-question eval** with both modes side-by-side to actually quantify the graph delta.

### Branch / git status
- Branch: `master`. Main branch for PRs: `main` (per repo CLAUDE setup).
- All changes in this session are uncommitted. Files modified:
  - `retriever.py` (Voikko, treaty boost, neighbor expansion, parent cap by type, graph load + `retrieve_with_graph`, caches)
  - `answerer.py` (planner treaty rules, `use_graph` parameter, traversal_log plumbing)
  - `eval_harness.py` (`--mode` flag, `graph_answer`, web-fallback default off for fair eval)
- Files created: `graph_builder.py`, `docs/agents/handoff/2026-05-23-001-graphrag-voikko-graph-layer.md`
- New data files (gitignored): `data/bm25.pkl`, `data/voikko_cache.json`, `data/graph.pkl`, `data/graph_edges.json`, `data/parent_nodes.json`

### Setup gotchas
- `libvoikko` is a native dep — install via `brew install libvoikko` BEFORE `pip install libvoikko`.
- `.venv` uses Python 3.14 (path: `/Users/ayaan/Projects/taxxa/.venv/bin/python3`).
- `.env` contains `OPENROUTER_API_KEY` — source it via `set -a && source .env && set +a` before any LLM-touching command.
- The `rank_bm25` package is not in `requirements.txt` (no requirements.txt exists — deps are inferred from `.venv` contents).
