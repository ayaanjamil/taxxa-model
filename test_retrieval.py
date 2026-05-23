"""
Free (no-LLM) retrieval diagnostic. Runs `retriever.diagnose` on the failing
TyEL query plus a few sanity checks.

Usage: python test_retrieval.py
"""

from retriever import diagnose, retrieve, expand_query

CASES = [
    {
        "name": "TyEL contribution age limit by birth year (the failing case)",
        "query": (
            "A Finnish employee was born in 1958. They are 67 years old and still "
            "in active employment. Are they still required to pay TyEL (statutory "
            "pension insurance) contributions, and until what age does the "
            "obligation continue for someone born in 1958 versus someone born in 1962?"
        ),
        "expected": "porrastetusti joko 68, 69 tai 70",
    },
    {
        "name": "TyEL acronym expansion sanity",
        "query": "TyEL maksuvelvollisuus syntymävuoden 1958 ja 1962 yläikäraja",
        "expected": "porrastetusti joko 68, 69 tai 70",
    },
]


def main():
    print("=" * 72)
    print("Acronym expansion check")
    print("=" * 72)
    for c in CASES[:1]:
        print(f"  IN:  {c['query'][:120]}...")
        print(f"  OUT: {expand_query(c['query'])[:160]}...")
    print()

    for c in CASES:
        print("=" * 72)
        print(c["name"])
        print(f"Query: {c['query'][:100]}...")
        print(f"Expected substring: {c['expected']!r}")
        diagnose(c["query"], c["expected"], top_k=30)
        print()


if __name__ == "__main__":
    main()
