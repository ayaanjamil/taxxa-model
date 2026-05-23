# Build Journey

## Phase 1 — Setup
- Reviewed plan, confirmed layered approach (baseline → graph → agents)
- Downloaded `finland_kb` (63k HTML files, 584MB) + `question_bank.json` (83 QA pairs)
- Created Python venv, installed dependencies

## Phase 2 — Eval Harness
- Wrote `eval_harness.py` — loads QA bank, calls answer function, LLM-judges key fact coverage, prints scores by tier
- Confirmed baseline works: stub answer returns 0% across all tiers

## Phase 3 — Parser
- Wrote `parser.py` — keyword-filters to tax-relevant HTML, extracts title/text/§-refs/date into `nodes.json`
- Hit several issues: Finnish morphology breaking statute detection, § notation with case suffixes, vero files cut off by limit
- Fixed: stem-based statute matching, updated §-ref regex, vero processed before finlex
- Final corpus: 37,324 nodes (1,826 vero + 35,498 finlex)

## Phase 4 — Baseline Retriever + Answerer (broken)
- Wrote `retriever.py` (vector search + reference expansion) and `answerer.py` (LLM with retrieved context)
- First eval: 0% fact coverage, 93% citation rate. Citations were appearing but the facts were wrong.

## Phase 5 — Diagnosis & Repair
Found seven cascading bugs (see challenges.md #6–#12):
1. Reference expansion 0% resolution (path-based IDs vs `TVL_§33` references) — fixed with section index
2. Embedding gap on English queries vs Finnish corpus — fixed with planner-generated Finnish keywords + title-substring matching
3. Keyword-matched nodes buried by vector scores — fixed with score boost
4. Naive merge ranking irrelevant nodes above target — fixed with Reciprocal Rank Fusion
5. Planner echoing user compound words — fixed with base-stem instruction
6. format_nodes truncation cutting off key facts — raised text caps
7. Judge LLM JSON wrapped in markdown → silent parse error → all evals reported 0%

## Phase 6 — Eval (after fixes)
- Basic tier: 0% → 40% fact coverage (Q1, Q4, Q7, Q8 all PASS or partial PASS)
- Confirmed Q1 (pääomatulo 34%) and Q4 (avainhenkilö 25% for 2026, 32% before) cite the correct vero documents with verbatim text quotes
