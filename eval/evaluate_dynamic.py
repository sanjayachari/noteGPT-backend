
import json
import sys
from pathlib import Path

from app.db.session import SessionLocal
from app.services.retriever import retrieve_similar_chunks
from app.services.reranker import rerank_chunks


TOP_K = 5
CANDIDATE_K = 20


def calculate_metrics(ranks):
    total = len(ranks)

    if total == 0:
        return {
            "recall": 0,
            "mrr": 0,
        }

    recall = sum(1 for rank in ranks if rank is not None) / total

    reciprocal_ranks = [
        1 / rank if rank is not None else 0
        for rank in ranks
    ]

    mrr = sum(reciprocal_ranks) / total

    return {
        "recall": recall,
        "mrr": mrr,
    }


def evaluate(document_id: int):
    dataset_file = (
        Path(__file__).parent
        / "datasets"
        / f"document_{document_id}.json"
    )

    if not dataset_file.exists():
        print(f"Dataset not found: {dataset_file}")
        return

    with open(dataset_file, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    db = SessionLocal()

    baseline_ranks = []
    reranked_ranks = []

    try:
        for index, item in enumerate(dataset, start=1):
            question = item["question"]
            expected_ids = set(item["expected_chunk_ids"])

            # 1. Baseline vector retrieval
            baseline_chunks = retrieve_similar_chunks(
                db=db,
                document_id=document_id,
                query=question,
                top_k=TOP_K,
            )

            baseline_rank = next(
                (
                    rank
                    for rank, chunk in enumerate(baseline_chunks, start=1)
                    if chunk.id in expected_ids
                ),
                None,
            )

            # 2. Retrieve candidates for reranking
            candidates = retrieve_similar_chunks(
                db=db,
                document_id=document_id,
                query=question,
                top_k=CANDIDATE_K,
            )

            # 3. Rerank candidates using Cross-Encoder
            reranked_results = rerank_chunks(
                query=question,
                chunks=candidates,
                top_k=TOP_K,
            )

            reranked_rank = next(
                (
                    rank
                    for rank, (chunk, score) in enumerate(
                        reranked_results,
                        start=1,
                    )
                    if chunk.id in expected_ids
                ),
                None,
            )

            baseline_ranks.append(baseline_rank)
            reranked_ranks.append(reranked_rank)

            print(f"\nQuestion {index}/{len(dataset)}")
            print(question)
            print(f"Expected chunk IDs: {sorted(expected_ids)}")
            print(f"Baseline rank: {baseline_rank}")
            print(f"Reranked rank: {reranked_rank}")

        # 4. Calculate final metrics
        baseline_metrics = calculate_metrics(baseline_ranks)
        reranked_metrics = calculate_metrics(reranked_ranks)

        print("\n" + "=" * 50)
        print("DYNAMIC RETRIEVAL EVALUATION")
        print("=" * 50)

        print(f"Document ID: {document_id}")
        print(f"Total questions: {len(dataset)}")

        print("\nBASELINE VECTOR SEARCH")
        print(f"Recall@{TOP_K}: {baseline_metrics['recall']:.2%}")
        print(f"MRR@{TOP_K}: {baseline_metrics['mrr']:.4f}")

        print("\nCROSS-ENCODER RERANKING")
        print(f"Recall@{TOP_K}: {reranked_metrics['recall']:.2%}")
        print(f"MRR@{TOP_K}: {reranked_metrics['mrr']:.4f}")

    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m eval.evaluate_dynamic <document_id>")
        sys.exit(1)

    evaluate(int(sys.argv[1]))
