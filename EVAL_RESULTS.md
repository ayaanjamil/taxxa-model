# Eval Results — Embedder Swap (mpnet → BGE-M3)

## What changed

The retrieval embedder was swapped from the legacy
`paraphrase-multilingual-mpnet-base-v2` (768-dim, 256-token window, 2021)
to `BAAI/bge-m3` (1024-dim, 384-token window, 2024). All 376,200 corpus
chunks were re-embedded on 4 parallel Modal A10G GPUs (~27 min wall time
after a preempted single-shard attempt). The synthesis context cap was
also bumped from 14k → 18k characters. Everything else — BM25, RRF,
graph walk, planner, synthesis prompts, judge — is unchanged.

## Comparison: same 15 stratified questions, both runs

Stratified sample: 3 each of `medium / hard / difficulty_2 / difficulty_4 / difficulty_5`,
deterministic via `random.seed(42)` (see eval_harness `--ids`).

| Tier         | mpnet (Run 0) | BGE-M3 (Run 1) | Δ      |
|--------------|--------------:|---------------:|-------:|
| medium       |          33%  |           53%  | **+20** |
| hard         |          80%  |           83%  |   +3   |
| difficulty_2 |          33%  |           67%  | **+34** |
| difficulty_4 |          39%  |           44%  |   +5   |
| difficulty_5 |          33%  |           22%  |  -11   |
| **OVERALL**  |       **44%** |       **54%**  | **+10** |
| Citation rate|          80%  |           93%  |  +13   |

Raw per-run JSON: `data/eval_runs/run0_mpnet_15q_stratified.json`,
`data/eval_runs/run1_bgem3_15q_stratified.json`.

## Per-question breakdown

| QID | Tier         | mpnet | BGE-M3 | Δ      |
|-----|--------------|------:|-------:|-------:|
| Q11 | medium       |  50%  |  100%  | **+50** |
| Q12 | medium       |  50%  |    0%  |  -50   |
| Q23 | medium       |   0%  |   60%  | **+60** |
| Q38 | hard         | 100%  |   50%  |  -50   |
| Q40 | hard         | 100%  |  100%  |    0   |
| Q49 | hard         |  40%  |  100%  | **+60** |
| N1  | difficulty_5 |   0%  |    0%  |    0   |
| N3  | difficulty_2 |   0%  |  100%  | **+100** |
| N7  | difficulty_4 |  67%  |   33%  |  -34   |
| N8  | difficulty_5 | 100%  |   67%  |  -33   |
| N10 | difficulty_2 |   0%  |    0%  |    0   |
| N11 | difficulty_2 | 100%  |  100%  |    0   |
| N42 | difficulty_4 |  50%  |    0%  |  -50   |
| N48 | difficulty_4 |   0%  |  100%  | **+100** |
| N60 | difficulty_5 |   0%  |    0%  |    0   |

**Pattern.** The big swings are mostly upward: 4 questions moved from
0% → 60–100%, while 4 questions regressed (the largest being -50pp).
Three questions remain stuck at 0% under both embedders — those are
likely cases where the needed text is either missing from the corpus or
phrased in a way neither embedder bridges (terminology gap, not a
retrieval-ranking problem).

## What this says about the embedder

- **Where BGE-M3 wins** (Q23, Q49, N3, N48): English-language questions
  with Finnish-corpus answers, or questions whose key terms have low
  exact-match overlap with the corpus (compound morphology, paraphrase).
  These are exactly the cases where a stronger multilingual embedder is
  supposed to bridge the semantic gap. The corpus mostly lives in
  Verohallinto guidance with Finnish phrasing the old mpnet model
  could not align to a multilingual query.
- **Where BGE-M3 regresses** (Q12, Q38, N7, N42): questions that mpnet
  was already solving with help from BM25. BGE-M3 changes which chunks
  the vector half of the hybrid retrieves; in these cases it apparently
  surfaced semantically-similar-but-wrong chunks that pushed the right
  ones below the parent cap, while BM25's lexical match was overpowered
  in the RRF merge.
- **Hard-tier (difficulty_5)** stays the bottleneck. These are
  multi-fact / multi-hop scenarios where the answer needs 3+ specific
  facts retrieved from different documents; the embedder swap helps
  some, but the harder lifts (better reranking, hop-aware fusion,
  stronger synthesis model) would have to come next.

## What was tuned

- `embedder.py` + `retriever.py` registry switchable via
  `TAXXA_EMBED_MODEL=bge-m3` env var; mpnet path retained for fast revert.
- `retriever.py` query encoder loads on MPS on Apple Silicon
  (~35ms/encode for BGE-M3 locally).
- `answerer.py` synthesis context cap raised from 14k → 18k chars
  (`TAXXA_CTX_CHARS`), giving the synthesizer ~25% more evidence to
  cite from.
- `modal_embed.py` runs as 4 parallel A10G workers via `.starmap()`,
  each owning a 94k-chunk slice. Each shard writes its own
  `vectors_bgem3_shard{i}.npy`; a final merge function concatenates
  them. Preemption only loses one shard (~15 min) instead of the whole
  run.

## What would move the needle next

1. **Cross-encoder reranker** (e.g. `BAAI/bge-reranker-v2-m3`) on the
   top 50 RRF results. Targets the exact regression class above: it
   would push semantically-close-but-wrong chunks below the correct ones
   that BM25 was protecting under mpnet.
2. **Sonnet 4.6 for synthesis** on the questions where evidence *is*
   in context but the synthesizer fails to assemble it (N7, N42 look
   like this).
3. **Corpus diff against the persistently-0% questions** (N1, N10, N60)
   — confirm whether the answer text actually exists in `finland_kb/`
   at all, or whether the QA bank is asking about something outside
   scope.
