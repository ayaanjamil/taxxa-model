# Handoff: Cheap-Mode Env Config, Streaming FastAPI Backend, and Next.js Cytoscape Hookup

**Date:** 2026-05-23
**Branch:** `master` (no commits made this session — all changes uncommitted)
**Prior handoff:** `2026-05-23-001-graphrag-voikko-graph-layer.md`
**Plan file:** `/Users/ayaan/.claude/plans/look-at-docs-agents-handoff-and-iterative-gadget.md` (overwritten three times during the session — final state covers the streaming work)
**Repos touched:**
- `/Users/ayaan/Projects/taxxa` (Python backend)
- `/Users/ayaan/Projects/taxxa-frontend` (Next.js 16 + React 19 frontend)

---

## What was accomplished

### 1. Env-controlled model config ("cheap mode")

Replaced hardcoded model constants so an `.env.cheap` overlay can swap in cheaper OpenRouter models for the planner + judge while keeping Claude Haiku for the answer-synthesis call (where Finnish quality matters most).

Edits:
- `answerer.py:19-21` — `ANSWER_MODEL`, `PLANNER_MODEL`, `WEB_FALLBACK_MODEL` now read `TAXXA_ANSWER_MODEL` / `TAXXA_PLANNER_MODEL` / `TAXXA_WEB_MODEL` env vars with the current Anthropic / Perplexity defaults preserved.
- `eval_harness.py:21` — `JUDGE_MODEL` reads `TAXXA_JUDGE_MODEL`.

New files:
- `.env.cheap` — sets `TAXXA_PLANNER_MODEL=deepseek/deepseek-chat-v3.1` and `TAXXA_JUDGE_MODEL=deepseek/deepseek-chat-v3.1`. Intentionally leaves `TAXXA_ANSWER_MODEL` unset (Haiku stays). User can flip the answer model after a small A/B confirms Finnish quality holds.
- `scripts/run_cheap_eval.sh` — runs both eval modes back-to-back with cheap-mode env loaded, streams output via `python3 -u | tee`, and saves both `.log` and `.json` outputs to `data/eval_runs/<timestamp>_{baseline,graph}.{log,json}`. Forwards extra CLI args via `"$@"` so `./scripts/run_cheap_eval.sh --sample 20 --seed 42` works.

Verified: `import os; os.environ['TAXXA_PLANNER_MODEL']='deepseek/deepseek-chat-v3.1'; from answerer import ANSWER_MODEL, PLANNER_MODEL` returns the expected (Haiku, DeepSeek) pair.

### 2. First A/B eval ran — **comparison invalidated by mismatched samples**

User ran `./scripts/run_cheap_eval.sh --sample 20` (no `--seed`). The harness re-randomized between baseline and graph runs, so the two modes saw **different question subsets** (baseline: 2 hard / 5 medium; graph: 6 hard / 3 medium). Headline numbers were:
- Baseline: 60% fact coverage, 95% citation rate (n=20)
- Graph:    56% fact coverage, 85% citation rate (n=20)

Do not treat this as a real comparison. The fix is to pass `--seed 42` (or any fixed seed) to both runs so they sample the same questions. This was deferred — user said "lets do it next time."

Worth following up on once samples line up: **citation rate dropping** on graph mode (95% → 85%) showed up in 3 specific questions (N56, N28, Q47). Could be a real pattern — synthesis model not citing graph-expanded chunks — or sample noise.

### 3. FastAPI SSE backend (`api.py`) — new file

Single endpoint: `POST /ask` returns `text/event-stream`. Emits six event types in order:

| event     | payload                                                      | when                                          |
|-----------|--------------------------------------------------------------|-----------------------------------------------|
| `plan`    | `{sub_questions: [...]}`                                     | after planner returns                         |
| `entry`   | `{nodes: [{id, type, label, superseded, desc}]}`             | after retrieval (and again for any hop endpoint not in the entry set) |
| `hop`     | `{from, to, relation}`                                       | one per traversal edge, paced server-side     |
| `sources` | `{sources, hops, nodes, time_ms}`                            | after retrieval completes, before synthesis   |
| `token`   | `{text}`                                                     | streamed from `openai.chat.completions.create(stream=True)` |
| `done`    | `{time_ms}`                                                  | final                                         |

Request body (`AskRequest`):
- `question: str`
- `use_graph: bool = True`
- `hop_delay_ms: int = 300`
- `top_k: int = 10`
- `max_hops: int = 30` — **critical** — caps the traversal animation. Without this, smoke tests emitted 252 hops for a single question (75s of animation at 300ms pacing).

Lifespan handler calls `_load()` + `_load_graph()` on startup so the first request doesn't pay the ~30s cold load. Server prints `Ready.` when warm.

Reuses (no re-implementation):
- `_plan` from `answerer.py:132`
- `retrieve_with_graph` from `retriever.py:306`
- `retrieve` from `retriever.py:191`
- `format_nodes` from `retriever.py:403`
- `SYNTHESIS_SYSTEM`, `ANSWER_MODEL`, `_get_client` from `answerer.py`
- `_nodes` dict from `retriever.py` (for resolving graph-hop endpoints back to chunks)

Internal helpers in `api.py`:
- `_to_graph_node(chunk)` — chunk dict → frontend `GraphNode` shape. Picks `type` from `node.type`, falls back to `GUIDANCE` for vero / `ARTICLE` for finlex. `superseded` from `superseded_by`. `label` is `§{section} {statute}` when available, else first 38 chars of title.
- `_to_source(gn, node)` — `GraphNode` + original chunk → frontend `Source` shape (with `dotType` / `tag` / `tagLabel` for the right-side panel).
- `_dedupe_by_parent(chunks)` — first-occurrence dedupe across the chunks returned by all sub-questions.

Install: `fastapi` + `uvicorn[standard]` were `pip install`'d into the `.venv`. Not in any requirements file (handoff #001 notes none exists).

Run command: `set -a && source .env && source .env.cheap && set +a && .venv/bin/python3 -m uvicorn api:app --port 8000 --log-level info`

Smoke test: `curl -N -s -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' -d '{"question":"...","use_graph":true,"hop_delay_ms":50}' | grep -E "^event:" | sort | uniq -c` returns the expected 1×plan / 1×entry / N×hop / 1×sources / N×token / 1×done.

CORS: `localhost:3000` and `127.0.0.1:3000` whitelisted. Adjust if frontend is ever on a different port (Next.js will fall back to `:3001` if `:3000` is taken — see "Gotchas" below).

### 4. Frontend wiring (`taxxa-frontend`)

The frontend already had the right scaffolding when this session started: Cytoscape was installed, mock data in `src/mock-data/{graph,answers}.ts` defined the exact shapes the backend now emits, and `src/store/query-store.ts` had a `setTimeout`-based mock of the streaming UX. **The frontend's mock data shape was treated as the contract** — the backend was shaped to match, not the other way around.

New files:
- `src/lib/sse.ts` — `streamSSE(url, body, signal)` async generator. POST + ReadableStream reader (`EventSource` is GET-only). ~40 lines, no extra dependency. Handles multi-line `data:` fields and JSON parsing.
- `src/lib/citations.ts` — `parseToParagraphs(rawText)` splits the streaming buffer on `\n\n+` into paragraphs and runs a `[id]` regex to split each into alternating `text` / `cite` spans. Safe to call on every token (sub-ms on <2KB answers).

Rewritten:
- `src/store/query-store.ts` — `sendMessage` is now an `async` function that consumes `streamSSE`. Per-event handlers mutate the assistant message in `messages[]`:
  - `plan` → stores `subQuestions` on the assistant message (not yet rendered, available for future UI)
  - `entry` → merges new nodes into `graph.nodes` (dedupes by id)
  - `hop` → appends an edge to `graph.edges` (no node creation — backend pre-emits an `entry` event for any new hop endpoint)
  - `sources` → sets `answer.sources`, `answer.hops`, `answer.nodes`, `answer.timeMs`
  - `token` → appends to `rawText` buffer and re-parses into `answer.paragraphs` via `parseToParagraphs`
  - `done` → flips `loading: false`
- An `AbortController` is stored on the state; sending a new question aborts any in-flight stream.
- API URL is configurable via `NEXT_PUBLIC_TAXXA_API` (defaults to `http://localhost:8000/ask`).
- The hardcoded initial message pair was removed — the thread starts empty.
- `AssistantMessage` interface gained two fields: `rawText: string` (token buffer) and `subQuestions?: string[]`.

Modified:
- `src/components/panels/chat-panel.tsx:34` — `AssistantBubble` skeleton now hides as soon as the first token OR source arrives (`loading && !hasContent`), not only when `loading` flips false. Without this change the user would see a static skeleton for the entire stream.
- `src/components/panels/graph-panel.tsx` — **largest frontend change**. Previously destroyed and rebuilt the entire Cytoscape instance on every `graphData` prop change. That would have flashed and re-laid-out the graph on every hop. Rewrote to:
  - Initialize Cytoscape **once** (gated on `ResizeObserver` reporting a non-zero container).
  - Maintain two `Ref<Set<string>>` of seen node IDs and seen edge IDs.
  - On each `graphData` update, diff against the seen sets, add only new elements via `cy.add(...)`, re-run a `cose` layout with `animate: true, animationDuration: 400`.
  - Newly added elements get a `.newly-added` class (purple highlight) for 600ms so the user sees what just appeared.
  - Includes a "reset" heuristic that clears Cytoscape state when a brand-new question's first event arrives (detected via shrinking node set).

Frontend typecheck (`npx tsc --noEmit`) passes clean after all changes.

### 5. Tunable demo knobs

In `query-store.ts:sendMessage` the SSE request body sets `hop_delay_ms: 300` and `max_hops: 30`. These are the two main "demo dramaturgy" knobs:
- Drop `hop_delay_ms` to ~100 for a more honest pace; raise to 500+ for a slower, more dramatic walk.
- `max_hops: 30` is the watchability cap. Without it, complex queries (e.g. capital-income tax) emit 250+ hops because one popular vero parent has 20+ outgoing `cites` edges.

---

## Key decisions

1. **SSE over WebSocket.** One-way server→client streaming; no need for the client to send anything mid-stream. Simpler FastAPI surface (`StreamingResponse` + `text/event-stream`) and zero extra frontend dep.

2. **POST + ReadableStream + manual SSE parsing on the frontend.** `EventSource` is GET-only. Question payloads are JSON with multiple fields (`use_graph`, `hop_delay_ms`, `max_hops`), so POST is the right verb. Wrote ~40 lines of parser instead of adding `@microsoft/fetch-event-source`.

3. **Artificial hop pacing on the server** (`asyncio.sleep(hop_delay_ms / 1000)` between yields) instead of dumping all hops and pacing client-side. User explicitly chose this in the planning Q&A: it's simpler, the client just renders what arrives, and pacing is centrally controlled per request.

4. **`max_hops` cap.** Without it the demo would have been a 75-second blur on common questions. Truncates `traversal_log` to the first N entries (graph-walk BFS order, so first hops are closest to entry parents and most relevant).

5. **Backend emits intermediate `entry` events for graph-walk endpoints.** When a hop's `to` or `from` parent wasn't in the original entry set, the server looks up any chunk for that parent in `_nodes` and emits a new `entry` event with just that node — *then* emits the `hop` event. This means the frontend never has to draw an edge to a node it hasn't seen, so the diff logic in `graph-panel.tsx` doesn't need to invent nodes on the fly.

6. **Re-parse the entire token buffer on every token.** Considered incremental parsing but the buffer is <2KB and `parseToParagraphs` is sub-ms. Not worth the complexity. If answers grow to >10KB, revisit.

7. **Kept Claude Haiku as the answer model under cheap mode.** DeepSeek V3.1 is the planner/judge default, but the Finnish synthesis call still uses Haiku. Citation discipline and Finnish-legal-prose quality are the riskiest swap; gate it behind an A/B before flipping. The env var is wired — flipping is one line in `.env.cheap`.

8. **Did NOT implement these (deferred from earlier plan):**
   - **Verifier loop** (`_verify` gated on `TAXXA_VERIFY`). Listed in handoff #001's "what plan says is next." Skipped because it adds 3s+ per answer and we're already trying to *reduce* latency. Revisit only if eval shows hallucinated citations.
   - **Iterative search loop** (replace one-shot planner with search→evaluate→refine). Listed as a user-suggested follow-up. Quality lever, not a latency lever — defer until after streaming is shipped and full eval has run.
   - **The latency-reduction plan** (parallelize sub-question retrieval, shrink synthesis context, prompt caching). Drafted as a 6-step plan during the session but **not executed** — user pivoted to streaming first. Still in the plan file's git history if needed.
   - **Frontend full visual polish.** The store + Cytoscape progressive rendering work; the `chat-panel` shows partial streamed text. Did not iterate on the visual experience (no end-to-end browser screenshot was taken this session — the dev server was already running on :3000 with the user's existing process, and the new code was expected to HMR in).

---

## Important context for future sessions

### File locations (additions this session)

- `/Users/ayaan/Projects/taxxa/api.py` — new (~200 lines)
- `/Users/ayaan/Projects/taxxa/.env.cheap` — new
- `/Users/ayaan/Projects/taxxa/scripts/run_cheap_eval.sh` — new
- `/Users/ayaan/Projects/taxxa-frontend/src/lib/sse.ts` — new
- `/Users/ayaan/Projects/taxxa-frontend/src/lib/citations.ts` — new
- `/Users/ayaan/Projects/taxxa-frontend/src/store/query-store.ts` — rewritten (was mock setTimeout, now SSE consumer)
- `/Users/ayaan/Projects/taxxa-frontend/src/components/panels/graph-panel.tsx` — rewritten (was full re-init, now incremental `cy.add`)
- `/Users/ayaan/Projects/taxxa-frontend/src/components/panels/chat-panel.tsx` — small edit (skeleton hides on first content)
- `/Users/ayaan/Projects/taxxa/answerer.py:19-21` — env-wrapped model constants
- `/Users/ayaan/Projects/taxxa/eval_harness.py:21` — env-wrapped judge model

### Eval results saved

`data/eval_runs/baseline_20260523_210834.{log,json}` and `data/eval_runs/graph_20260523_210834.{log,json}` — the mismatched-samples run from this session. Useful for *baseline-vs-graph diffing only after re-running with `--seed`*. Header tier breakdown is in the logs.

### Branch / git status

- Branch: `master`. Main branch for PRs: `master` (the project CLAUDE says `main` but the repo only has `master`).
- **Everything in this session is uncommitted.** Combined with the uncommitted changes from session #001 (Voikko, graph layer, retrieval fixes), that's a substantial uncommitted surface. Next session should commit before adding more.
- `docs/agents/handoff/` is untracked entirely (it was created in session #001 and is in `?? docs/` status per the original git snapshot).

### Setup gotchas

1. **Two Next.js dev servers can't coexist on :3000.** Your existing `next dev` process (PID 7687 at session time) was running on :3000. When the session tried to start a fresh one it landed on :3001 and immediately exited because Next.js detects the duplicate. CORS in `api.py` whitelists only `:3000` — if a future session opens the app on `:3001`, it must either kill the existing dev server or extend the CORS allowlist.

2. **HMR vs hard reload.** Changes to `query-store.ts` and the panel components should HMR, but zustand store changes are sometimes flaky on HMR. If the page seems to use the old mock data after a change, `Cmd+Shift+R` to hard reload.

3. **Env loading.** Both `.env` and `.env.cheap` must be sourced before starting uvicorn or the cheap-mode swap doesn't apply. The `scripts/run_cheap_eval.sh` script does this; for the API server you must do it manually: `set -a && source .env && source .env.cheap && set +a && .venv/bin/python3 -m uvicorn api:app --port 8000`.

4. **uvicorn `--reload` and the cold load.** Don't use `--reload` for demos. Every file save re-loads the 1.1 GB vector array, 507 MB nodes file, and 446 MB BM25 pickle. Use `--reload` for development only, then restart without it for the actual demo.

5. **First request can be slow even after warm-up** if you bumped `max_tokens` for answer synthesis or if DeepSeek is rate-limited that day. OpenRouter occasionally injects 5–10s latency on cheap models.

### Known issues / unfinished

- **No end-to-end browser verification this session.** The backend was curl-tested and the frontend was typecheck-tested, but nobody opened the app in a browser and asked Q20. Highest-priority test for the next session.
- **Citation parser is brittle on edge cases.** `parseToParagraphs` regex is `/\[([^\]]+)\]/g` — matches anything between brackets. A literal `[note]` in source text would be mis-parsed as a citation. Acceptable for the demo; worth tightening if it shows up in eval.
- **Graph reset heuristic in `graph-panel.tsx`.** Detects "new question" by checking if the incoming node count is smaller than the seen set. Robust for the typical case but could miss-fire in edge cases (e.g. an entry event with zero new nodes mid-stream). Tested only via typecheck, not at runtime.
- **`AssistantMessage.subQuestions` is never rendered.** Stored in state by the `plan` event handler but no UI consumes it. Easy ~10-line addition to `chat-panel.tsx` if a future session wants to display the agentic planning step.
- **No abort UI.** The store stores an `AbortController` and aborts on a new question, but there's no "stop generating" button. Cheap add: surface `abortController.abort()` from the store.
- **Full 83-question eval still not run** with matched samples. The `--seed` fix is one CLI arg away.

### What the plan file says is next (and is NOT done)

The plan file at `/Users/ayaan/.claude/plans/look-at-docs-agents-handoff-and-iterative-gadget.md` was rewritten three times during this session. The final state describes the streaming work that was just completed. Earlier states (in git history of `~/.claude/plans/` if it's git-tracked, otherwise gone) covered:
- **Cheap-mode setup** — done
- **Latency reduction** (parallelize retrieval, shrink context, prompt caching, streaming) — only streaming was done; the parallelize/shrink/caching items remain
- **Frontend hookup** — done

Real next steps from the original `plan.md` Phase 6 perspective:
1. **Run the full 83-question eval with `--seed 42`** under both modes for a real baseline-vs-graph delta. This is the data the README needs.
2. **Commit everything.** Sessions #001 and #002 together are a large uncommitted surface.
3. **Browser-test end-to-end** with Q20 (Finland–Austria treaty). Confirm hops animate, citations render, answer streams.
4. **Latency reduction** — parallelize sub-question retrieval (`answerer.py:222`), shrink `format_nodes(max_chars=20000 → 10000)`, lower `max_tokens=1000 → 600`, enable Anthropic prompt caching on `SYNTHESIS_SYSTEM`.
5. **README** with the full eval table and a one-paragraph design narrative — explicitly judge-weighted per the original `plan.md`.
