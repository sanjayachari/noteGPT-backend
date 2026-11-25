
import json
from pathlib import Path

from app.db.session import SessionLocal
from app.services.retriever import retrieve_similar_chunks
from app.services.reranker import rerank_chunks


DATASET_PATH = Path(__file__).parent / "retrieval_dataset.json"

DOCUMENT_ID = 4
TOP_K = 5
CANDIDATE_K = 20


def calculate_reciprocal_rank(retrieved_pages, expected_pages):
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected_pages:
            return 1 / rank

    return 0


def evaluate():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    total = len(dataset)
    db = SessionLocal()

    baseline_hits = 0
    reranked_hits = 0

    baseline_rr_total = 0
    reranked_rr_total = 0

    try:
        for item in dataset:
            question = item["question"]
            expected_pages = item["expected_pages"]

            # 1. Baseline retrieval: MiniLM retrieves top 5
            baseline_chunks = retrieve_similar_chunks(
                db=db,
                document_id=DOCUMENT_ID,
                query=question,
                top_k=TOP_K,
            )

            # 2. Retrieve candidates for reranking
            candidates = retrieve_similar_chunks(
                db=db,
                document_id=DOCUMENT_ID,
                query=question,
                top_k=CANDIDATE_K,
            )

            # 3. Cross-Encoder reranks candidates
            reranked_results = rerank_chunks(
                query=question,
                chunks=candidates,
                top_k=TOP_K,
            )

            reranked_chunks = [
                chunk for chunk, score in reranked_results
            ]

            # 4. Extract page numbers
            baseline_pages = [
                chunk.page_number for chunk in baseline_chunks
            ]

            reranked_pages = [
                chunk.page_number for chunk in reranked_chunks
            ]

            # 5. Calculate Recall@5 hits
            baseline_hit = any(
                page in expected_pages for page in baseline_pages
            )

            reranked_hit = any(
                page in expected_pages for page in reranked_pages
            )

            baseline_hits += baseline_hit
            reranked_hits += reranked_hit

            # 6. Calculate reciprocal rank
            baseline_rr = calculate_reciprocal_rank(
                baseline_pages, expected_pages
            )

            reranked_rr = calculate_reciprocal_rank(
                reranked_pages, expected_pages
            )

            baseline_rr_total += baseline_rr
            reranked_rr_total += reranked_rr

            # 7. Display individual results
            print("\n" + "=" * 60)
            print(f"Question: {question}")
            print(f"Expected pages: {expected_pages}")
            print(f"Baseline pages: {baseline_pages}")
            print(f"Reranked pages: {reranked_pages}")

            print(f"Baseline: {'PASS' if baseline_hit else 'FAIL'}")
            print(f"Reranked: {'PASS' if reranked_hit else 'FAIL'}")

            print("\nCross-Encoder scores:")

            for chunk, score in reranked_results:
                print(
                    f"Page {chunk.page_number}: {float(score):.4f}"
                )

    finally:
        db.close()

    # 8. Display final comparison AFTER all questions are evaluated
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")

    print(f"Total questions: {total}")
    print(f"Baseline Recall@5: {baseline_hits / total:.2%}")
    print(f"Reranked Recall@5: {reranked_hits / total:.2%}")

    print(f"Baseline MRR@5: {baseline_rr_total / total:.2f}")
    print(f"Reranked MRR@5: {reranked_rr_total / total:.2f}")


if __name__ == "__main__":
    evaluate()
