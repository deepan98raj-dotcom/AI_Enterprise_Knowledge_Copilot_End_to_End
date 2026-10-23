
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.retrieval import retrieve

CASES_FILE = ROOT / "data" / "evaluation" / "retrieval_cases.json"


def phrase_found(result, phrases):
    text = result.get("text", "").casefold()
    return any(
        phrase.casefold() in text
        for phrase in phrases
        if phrase.strip()
    )


def main():
    if not CASES_FILE.exists():
        print(f"Evaluation dataset not found: {CASES_FILE}")
        return

    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))

    if not cases:
        print("The evaluation file is empty. Add test cases first.")
        return

    hits = []
    reciprocal_ranks = []

    for case in cases:
        question = case["question"]
        expected = case.get("expected_phrases", [])

        if not expected:
            print(f"Skipping case without expected phrases: {question}")
            continue

        results = retrieve(question, top_k=5)
        relevant_rank = next(
            (
                rank
                for rank, result in enumerate(results, start=1)
                if phrase_found(result, expected)
            ),
            None,
        )

        hits.append(1 if relevant_rank else 0)
        reciprocal_ranks.append(
            1 / relevant_rank if relevant_rank else 0
        )

        print(f"\nQuestion: {question}")
        print(f"Relevant result in top 5: {relevant_rank is not None}")
        print(f"First relevant rank: {relevant_rank or 'Not found'}")

        if relevant_rank is None:
            print("\nTop retrieved passages for debugging:")

            for rank, result in enumerate(results, start=1):
                print(
                    f"\n[{rank}] "
                    f"Document: {result.get('document')} | "
                    f"Page: {result.get('page')} | "
                    f"Score: {result.get('score')}"
                )
                print(result.get("text", "")[:500])

    if not hits:
        print("No usable evaluation cases were found.")
        return

    print("\n--- Retrieval Evaluation ---")
    print(f"Evaluated questions: {len(hits)}")
    print(f"Hit@5: {sum(hits) / len(hits):.3f}")
    print(f"MRR@5: {sum(reciprocal_ranks) / len(reciprocal_ranks):.3f}")


if __name__ == "__main__":
    main()
