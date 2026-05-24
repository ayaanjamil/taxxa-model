"""
Answer generation: decompose question into sub-questions, retrieve in parallel, synthesize.

Flow:
  1. Planner: decompose question into 1-4 sub-questions (or use question as-is if simple)
  2. Retrieve: run vector search + graph expansion for each sub-question
  3. Merge: dedupe nodes, keep highest score per node
  4. Synthesize: single LLM call with all evidence
"""

import json
import os

from openai import OpenAI

from retriever import format_nodes, retrieve, retrieve_with_graph

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
ANSWER_MODEL = os.getenv("TAXXA_ANSWER_MODEL", "anthropic/claude-haiku-4-5")
PLANNER_MODEL = os.getenv("TAXXA_PLANNER_MODEL", "anthropic/claude-haiku-4-5")
WEB_FALLBACK_MODEL = os.getenv("TAXXA_WEB_MODEL", "perplexity/sonar")  # search + synthesis in one call

# Substrings that signal the corpus didn't contain the answer.
# Match against the lowercased answer; keep these short and unambiguous.
REFUSAL_MARKERS = (
    # English
    "cannot find",
    "i cannot find",
    "cannot provide",
    "cannot answer",
    "cannot fully answer",
    "unable to find",
    "unable to provide",
    "unable to answer",
    "not enough information",
    "insufficient information",
    "do not contain",
    "do not provide",
    "no information",
    # Finnish
    "en löydä",
    "valitettavasti, en löydä",
    "en pysty löytämään",
    "en löytänyt",
    "en voi vastata",
    "ei ole tietoa",
    "ei sisällä",
    "en pysty vastaamaan",
)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE,
        )
    return _client


PLANNER_SYSTEM = """You are a legal research planner for a Finnish tax law corpus.
The corpus is in Finnish. Return JSON {"sub_questions": ["...", "...", ...]}.

For each sub-question, generate a Finnish keyword search string using BASE Finnish stems
that would appear in actual tax documents. Do NOT echo back the user's compound words —
they are often longer than what the corpus uses.

CRITICAL: use base Finnish stems, not compound user terms:
- "pääomatulovero" → use "pääomatulo" (the base term in tax docs)
- "ennakonpidätys" → use "ennakonpidätys" or "pidätysprosentti"
- "avainhenkilöstatus" → use "avainhenkilö"
- "yleisradiovero" → use "yleisradiovero" or "Yle-vero" or "radiovero"
- "perintövero" → use "perintövero" or "perintöverotus"

Finnish tax term translations:
- key personnel / foreign specialist → avainhenkilö, ulkomailta tuleva palkansaaja
- withholding tax → lähdevero, lähdeveroprosentti
- tax card → verokortti, voimassaolo
- capital income → pääomatulo (NOT pääomatulovero)
- earned income → ansiotulo
- dividend → osinko
- corporate tax → yhteisövero
- VAT → arvonlisävero, ALV
- inheritance / gift tax → perintövero, lahjavero, PerVL
- commuting deduction → asunnon ja työpaikan väliset matkat, matkakuluvähennys
- broadcasting tax → yleisradiovero, Yle-vero
- pension insurance → TyEL, työeläkemaksu, työeläkevakuutusmaksu
- statute section → TVL §, AVL §, EPL §, VML §, PerVL §
- double tax treaty → verosopimus, kaksinkertaisen verotuksen välttäminen, sopimus [country]
- treaty dividend article → osinko lähdevero äänimäärä hallitsee prosenttia sopimusvaltio

TREATY QUESTIONS: When the question involves a bilateral tax treaty with a specific country,
EVERY sub-question MUST include the country name AND article-content keywords together
in the same query string. Treaty article texts (the actual paragraphs with the rates) do
NOT contain the country name themselves — only the title does. So you MUST combine them
in each query so BM25 matches both the title (country) and the article body (content).

For a Finland-Austria dividend treaty question, generate queries like:
- "Itävalta osinko äänimäärä hallitsee 10 prosenttia lähdevero" (combines country + Article 10 keywords)
- "Itävalta verosopimus osinko sopimusvaltio asuva yhtiö välittömästi" (combines country + paragraph 3 keywords)
- "Itävalta osinko kokonaismäärästä lähdevero prosenttia" (combines country + paragraph 2 keywords)
NEVER generate a sub-query that contains only article keywords without the country name —
that will match Article 10 of every bilateral treaty (Singapore, Kazakhstan, etc).

Generate 2-4 Finnish keyword strings, each targeting a specific fact.

Example input: "What is the withholding tax rate for a foreign specialist with key-personnel status?"
Example output: {"sub_questions": ["avainhenkilö lähdevero prosentti", "ulkomailta tuleva palkansaaja avainhenkilölaki verokortti voimassaolo"]}

Example input: "What is the capital income tax rate (pääomatulovero) above 30000 euros?"
Example output: {"sub_questions": ["pääomatulo veroprosentti 30000 ylittävä", "pääomatulo korotettu tuloveroprosentti TVL", "pääomatulosta suoritetaan veroa prosenttia"]}

Example input: "Under the Finland-Austria double tax treaty, what withholding rate applies to dividends?"
Example output: {"sub_questions": ["Itävalta osinko äänimäärä hallitsee 10 prosenttia lähdevero", "Itävalta verosopimus osinko sopimusvaltio asuva yhtiö välittömästi", "Itävalta osinko kokonaismäärästä lähdevero prosenttia 5"]}"""


SYNTHESIS_SYSTEM = """You are a Finnish tax law expert assistant. Answer based ONLY on the provided documents.

Rules:
- Every factual claim must cite the specific source using [node_id] notation.
- If the answer spans multiple documents, cite each one.
- If documents conflict on a rate or number, use the most recently dated one — it supersedes earlier versions.
- Prefer Verohallinto (vero) guidance over finlex amendment history when both cover the same fact.
- If you cannot find the answer in the provided documents, say so explicitly.
- Be precise with numbers, dates, and rates — do not round or approximate.
- Answer in the same language as the user's question. If the user explicitly requests a different language (e.g. "answer in Swedish"), use that language instead."""


def _plan(question: str) -> list[str]:
    """Decompose question into sub-questions. Falls back to [question] on any error."""
    try:
        resp = _get_client().chat.completions.create(
            model=PLANNER_MODEL,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": f"Question: {question}"},
            ],
            max_tokens=400,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        data = json.loads(raw)
        subs = data.get("sub_questions", [question])
        return subs if subs else [question]
    except Exception:
        return [question]


def _merge_nodes(node_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion: score(node) = sum_q 1/(k + rank_in_q).
    Standard approach for merging multiple ranked lists. Nodes appearing high in
    any sub-query get high score; multi-list nodes get higher score.
    """
    scores: dict[str, float] = {}
    seen: dict[str, dict] = {}
    for node_list in node_lists:
        for pos, n in enumerate(node_list):
            nid = n["id"]
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + pos)
            if nid not in seen:
                seen[nid] = n
    return sorted(seen.values(), key=lambda n: -scores[n["id"]])


def _looks_like_refusal(answer: str) -> bool:
    """Detect when the corpus didn't contain the answer."""
    low = answer.lower()
    return any(marker in low for marker in REFUSAL_MARKERS)


WEB_FALLBACK_SYSTEM = """You are a Finnish tax and accounting law expert assistant.
The user's question could not be answered from the primary Finnish tax-law corpus
(Finlex statutes + Verohallinto guidance). Search the web for authoritative sources
— prefer kirjanpitolautakunta.fi (KILA accounting standards), vero.fi, finlex.fi,
suomi.fi, and reputable Finnish accounting/legal publications.

Rules:
- Be precise with numbers, dates, and rates — do not round.
- Cite the URL of each source you used.
- Answer in the same language as the question.
- If you still cannot find authoritative information, say so explicitly."""


def _web_fallback(question: str) -> str:
    """Fall back to a web-search-enabled model when local retrieval refuses."""
    try:
        resp = _get_client().chat.completions.create(
            model=WEB_FALLBACK_MODEL,
            messages=[
                {"role": "system", "content": WEB_FALLBACK_SYSTEM},
                {"role": "user", "content": question},
            ],
            max_tokens=1000,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Web fallback failed: {e}"


def _local_answer(question: str, top_k: int = 12, use_graph: bool = False) -> tuple[str, list[dict]]:
    """Returns (answer_text, traversal_log). traversal_log is empty if use_graph=False."""
    sub_questions = _plan(question)
    all_queries = sub_questions if sub_questions else [question]

    traversal_log: list[dict] = []
    if use_graph:
        node_lists = []
        for sq in all_queries:
            chunks, hops = retrieve_with_graph(sq, top_k=top_k)
            node_lists.append(chunks)
            traversal_log.extend(hops)
    else:
        node_lists = [retrieve(sq, top_k=top_k) for sq in all_queries]

    merged = _merge_nodes(node_lists)
    context = format_nodes(merged, max_chars=20000)

    user_prompt = f"Question: {question}\n\nSub-questions researched: {sub_questions}\n\nDocuments:\n{context}"

    resp = _get_client().chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1000,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip(), traversal_log


def answer(question: str, top_k: int = 12, allow_web_fallback: bool = True, use_graph: bool = False) -> str:
    """Local-first answer with optional web-search fallback on refusal.

    Set allow_web_fallback=False to force corpus-only mode (useful for eval runs
    where you want to measure pure RAG performance).
    """
    local, _hops = _local_answer(question, top_k=top_k, use_graph=use_graph)
    if not allow_web_fallback or not _looks_like_refusal(local):
        return local

    web = _web_fallback(question)
    return (
        f"{web}\n\n---\n"
        "*[Web-sourced — corpus did not contain this information.]*"
    )


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is the capital income tax rate for income exceeding 30,000 euros?"
    print(f"Q: {q}\n")
    print(answer(q))
