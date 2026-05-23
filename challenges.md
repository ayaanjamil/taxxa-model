# Build Challenges

## 1. Corpus scale — 63k files, 584MB
**Problem:** `finland_kb` contains 63,661 HTML files (541MB finlex + 44MB vero). Embedding everything naively would take hours and significant API cost.
**Fix:** Keyword filter on filename + first 2000 chars of content. Only files matching tax-relevant terms (vero, tulo, AVL, TVL, etc.) are parsed and embedded.

## 2. Finnish morphological declension breaks regex
**Problem:** Finnish inflects nouns by case. "tuloverolaki" (nominative) becomes "tuloverolain" (genitive), "tuloverolakia" (partitive), etc. Statute inference regex matched nominative form only — result: only 2 statutes detected from 2000 nodes (PerVL: 8, KiVL: 6).
**Fix:** Match word stems instead of full nominative forms. `tuloverola` matches all case forms of "tuloverolaki".

## 3. Finnish § notation with case suffixes
**Problem:** Finnish legal text writes section references as "12 §:n", "12 §:ään", "§:n 33" — the colon + case ending is grammatically required. Original REF_RE only matched bare "12 §" or "§ 12", giving 1% §-ref coverage.
**Fix:** Updated regex to match `\d+ §` and `§ \d+` — the number and § symbol are sufficient to identify a cross-reference regardless of the suffix.

## 5. Finlex corpus has no consolidated law sections
**Problem:** The `finlex/Laki/` files are amendment history files ("Laki X §:n muuttamisesta"), not consolidated law text. `Laki (säädöskokoelma)/Tuloverolaki.html` is a 60-char stub. TVL §124 (capital income tax rate) sits at char offset 63,283 in the amendment history file — completely past the 4000-char parse cap. Result: 0% fact coverage on basic questions despite citations being found.
**Fix:** Treat vero guidance as the primary fact source (it states current rates explicitly). Added 1.3× retrieval score boost for vero nodes and increased top_k from 8 to 12.

## 4. vero files never reached under --limit
**Problem:** `Path.rglob("*.html")` returns files in filesystem order — finlex (61,835 files) before vero (1,826 files). Running with `--limit 2000` always hit the cap inside finlex, leaving zero vero nodes parsed.
**Fix:** Explicitly walk vero first, then finlex. Vero is always fully parsed before the limit can cut it off.

## 6. Reference expansion was 0% effective
**Problem:** Each node's `references` list contains entries like `TVL_§33` extracted from text. But the actual node IDs are path-based (`finlex_Laki_Tuloverolaki_...`). 0 of 5,551 references matched any node ID — the reference expansion in `retrieve()` was completely dead.
**Fix:** Built a section index at load time in `retriever.py`. For each node with a known statute, scan its title + text for `\d+ §` patterns and map `{statute}_§{section}` → list of node IDs. Now 94% of references resolve.

## 7. English/Finnish embedding gap
**Problem:** The multilingual sentence-transformer scores English query "key-personnel withholding" only ~0.36–0.55 against the Finnish "avainhenkilö lähdevero" nodes, while the top-12 threshold is ~0.55–0.62. Domain-specific Finnish tax vocabulary doesn't bridge cleanly.
**Fix:** Two layers: (a) Planner LLM rewrites English questions into Finnish keyword strings before retrieval. (b) `_keyword_match_indices()` in `retriever.py` adds an exact-substring pass: scans all node titles first (high precision), then text previews, and force-includes matches into the candidate set with a score boost of `top_score * 1.5` so embedding noise doesn't drown them out.

## 8. Multi-query merging buried the right nodes
**Problem:** The planner generates 3-4 sub-queries that each get retrieved. The naive merge (keep first occurrence, sort by frequency) ranked irrelevant nodes that happened to appear in multiple sub-queries above highly relevant nodes that appeared in only one. For Q4 (avainhenkilö 2026 rate), the target node was at position 106 in the merged list — never made it into the LLM's 10k-char context.
**Fix:** Replaced with Reciprocal Rank Fusion in `answerer._merge_nodes()`: `score(node) = sum_q 1/(60 + rank_in_q)`. Standard approach; a node ranked #1 in any sub-query now beats junk that ranks #5 in three.

## 9. Planner echoed compound user words not present in corpus
**Problem:** Question says "pääomatulovero" (compound: capital-income-tax). The corpus actually uses the base "pääomatulo" (capital-income) in most contexts. Planner blindly copied "pääomatulovero" into Finnish sub-queries, retrieving nothing.
**Fix:** Updated planner system prompt with explicit "use base Finnish stems, not user compound terms" rule, with examples: pääomatulovero → pääomatulo, avainhenkilöstatus → avainhenkilö.

## 10. Original English question polluted multi-query retrieval
**Problem:** Including the user's English question in the retrieval pass alongside the planner's Finnish sub-queries added 5–10 high-vector-score noise nodes (KVL court rulings on US tax treaties etc.) that crowded out the Finnish keyword-matched results.
**Fix:** Drop the original question — use only the planner's sub-queries. The planner exists specifically to translate, so re-adding the source is double-counting noise.

## 11. format_nodes truncation cut off the answer
**Problem:** The avainhenkilö 2026 guidance has "25 prosentin suuruista lähdeveroa" at char 2149 of its text. With `body = n["text"][:2000]`, this got cut off; the LLM only saw the intro paragraph and answered "I cannot find the rate."
**Fix:** Raised per-node text cap from 800 → 2000 → 3500 chars in `format_nodes()`, total context cap from 6000 → 20000 chars in `answerer.answer()`. Claude Haiku 4.5 handles 20k context fine.

## 12. Judge LLM silent JSON parse failures
**Problem:** `judge_fact_coverage()` in `eval_harness.py` used `response_format={"type": "json_object"}` but Claude Haiku 4.5 via OpenRouter still wraps responses in ```json ... ``` markdown fences. Every `json.loads(content)` raised, defaulting to `{"covered": False}`. Result: even when answers contained the correct facts verbatim, the eval reported 0% fact coverage.
**Fix:** Strip leading ```json fences before `json.loads`. Surface the raw text in the reason field for debuggability. After fix, basic-tier coverage jumped from 0% → 40% on the same answers.
