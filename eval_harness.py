"""
Eval harness for Taxxa Finland QA Bank V1.
Usage:
  python eval_harness.py                  # run all 83 questions
  python eval_harness.py --sample 10      # run 10 random questions
  python eval_harness.py --dry-run        # skip LLM judge, print structure only
  python eval_harness.py --tier basic     # filter by tier
"""

import argparse
import json
import os
import random
import sys
from typing import Callable

from openai import OpenAI

QUESTION_BANK = "question_bank.json"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
JUDGE_MODEL = os.getenv("TAXXA_JUDGE_MODEL", "anthropic/claude-haiku-4-5")  # cheap judge


def load_questions(tier: str | None = None) -> list[dict]:
    with open(QUESTION_BANK) as f:
        data = json.load(f)
    entries = data["entries"]
    if tier:
        entries = [e for e in entries if e["tier"] == tier]
    return entries


def stub_answer(question: str) -> str:
    return ""


def real_answer(question: str) -> str:
    from answerer import answer
    return answer(question, allow_web_fallback=False)


def graph_answer(question: str) -> str:
    from answerer import answer
    return answer(question, allow_web_fallback=False, use_graph=True)


def judge_fact_coverage(fact: str, answer: str, client: OpenAI, dry_run: bool) -> dict:
    if dry_run or not answer.strip():
        return {"covered": False, "reason": "dry-run or empty answer"}
    prompt = (
        f'Fact to check: "{fact}"\n'
        f'Answer to check: "{answer}"\n'
        "Does the answer contain this fact? "
        "Return JSON: {\"covered\": true/false, \"reason\": \"...\"}"
    )
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict fact-checker. Return JSON {\"covered\": true/false, \"reason\": \"...\"}. "
                    "The fact and answer may be in Finnish or English. Evaluate semantically, not lexically. "
                    "A paraphrase counts; a contradiction does not."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=150,
    )
    raw = (resp.choices[0].message.content or "").strip()
    # Strip markdown code fences if the model wraps JSON in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception as e:
        return {"covered": False, "reason": f"parse error: {raw[:80]}"}


def has_citations(answer: str) -> bool:
    import re
    return bool(re.search(r"§\s*\d+|TVL|AVL|EPL|VML|KHO|KVL|VH-|ohje", answer, re.IGNORECASE))


def run_eval(
    answer_fn: Callable[[str], str],
    questions: list[dict],
    client: OpenAI,
    dry_run: bool,
    verbose: bool = False,
) -> dict:
    results = []
    tier_stats: dict[str, dict] = {}

    for i, q in enumerate(questions):
        qid = q["id"]
        tier = q["tier"]
        question = q["question"]
        key_facts = q.get("answer_key_facts", [q["answer"]])

        answer = answer_fn(question)
        cited = has_citations(answer)

        fact_results = []
        for fact in key_facts:
            verdict = judge_fact_coverage(fact, answer, client, dry_run)
            fact_results.append(verdict)

        covered_count = sum(1 for r in fact_results if r.get("covered", False))
        fact_score = covered_count / len(key_facts) if key_facts else 0.0

        entry = {
            "id": qid,
            "tier": tier,
            "fact_score": fact_score,
            "has_citation": cited,
            "facts_covered": covered_count,
            "facts_total": len(key_facts),
        }
        results.append(entry)

        if tier not in tier_stats:
            tier_stats[tier] = {"fact_scores": [], "citations": []}
        tier_stats[tier]["fact_scores"].append(fact_score)
        tier_stats[tier]["citations"].append(cited)

        status = "PASS" if fact_score >= 0.5 else "FAIL"
        print(f"[{i+1:02d}/{len(questions)}] {qid} ({tier}) — fact_score={fact_score:.0%} citation={cited} {status}")
        if verbose:
            print(f"  Q: {question[:80]}")
            print(f"  A: {answer[:120] if answer else '(empty)'}")
            for j, (fact, res) in enumerate(zip(key_facts, fact_results)):
                mark = "✓" if res.get("covered") else "✗"
                print(f"  {mark} fact[{j}]: {fact[:80]}")

    print("\n=== RESULTS BY TIER ===")
    overall_facts = []
    overall_cites = []
    for tier, stats in sorted(tier_stats.items()):
        avg_fact = sum(stats["fact_scores"]) / len(stats["fact_scores"])
        cite_rate = sum(stats["citations"]) / len(stats["citations"])
        n = len(stats["fact_scores"])
        print(f"  {tier:15s}  n={n:2d}  fact_coverage={avg_fact:.0%}  citation_rate={cite_rate:.0%}")
        overall_facts.extend(stats["fact_scores"])
        overall_cites.extend(stats["citations"])

    overall_fact = sum(overall_facts) / len(overall_facts) if overall_facts else 0
    overall_cite = sum(overall_cites) / len(overall_cites) if overall_cites else 0
    print(f"\n  {'OVERALL':15s}  n={len(overall_facts):2d}  fact_coverage={overall_fact:.0%}  citation_rate={overall_cite:.0%}")

    return {"results": results, "overall_fact_coverage": overall_fact, "overall_citation_rate": overall_cite}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Reproducible --sample selection")
    parser.add_argument("--ids", type=str, default=None,
                        help="Comma-separated question IDs to run (overrides --sample)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tier", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--mode", choices=["baseline", "graph"], default="baseline",
                        help="baseline = hybrid RAG; graph = graph-walk retrieval")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: set OPENROUTER_API_KEY or use --dry-run", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key or "dry-run", base_url=OPENROUTER_BASE)

    questions = load_questions(tier=args.tier)
    if args.ids:
        want = set(s.strip() for s in args.ids.split(",") if s.strip())
        questions = [q for q in questions if q["id"] in want]
        missing = want - {q["id"] for q in questions}
        if missing:
            print(f"WARNING: ids not found: {sorted(missing)}", file=sys.stderr)
    elif args.sample:
        if args.seed is not None:
            random.seed(args.seed)
        questions = random.sample(questions, min(args.sample, len(questions)))

    print(f"Running eval on {len(questions)} questions (mode={args.mode}, dry_run={args.dry_run})\n")

    if args.dry_run:
        answer_fn = stub_answer
    elif args.mode == "graph":
        answer_fn = graph_answer
    else:
        answer_fn = real_answer
    results = run_eval(answer_fn, questions, client, dry_run=args.dry_run, verbose=args.verbose)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
