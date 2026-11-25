
import json
import sys
from pathlib import Path

from app.db.session import SessionLocal
from app.services.retriever import retrieve_similar_chunks
from app.services.bm25_retriever import retrieve_bm25_chunks
from app.services.hybrid_retriever import retrieve_hybrid_chunks
from app.services.reranker import rerank_chunks


TOP_K = 5
CANDIDATE_K = 20


def find_rank(results, expected_ids):
    for rank, item in enumerate(results, start=1):
        chunk = item[0] if isinstance(item, tuple) else item

        if chunk.id in expected_ids:
            return rank

    return None


def calculate_metrics(ranks):
    total = len(ranks)

    if not total:
        return 0, 0

    recall = sum(rank is not None for rank in ranks) / total

    mrr = sum(
        1 / rank if rank is not None else 0
        for rank in ranks
    ) / total

    return recall, mrr


def evaluate(document_id):
    dataset_file = (
        Path(__file__).parent
        / "datasets"
        / f"document_{document_id}.json"
    )

    if not dataset_file.exists():
        print(f"Dataset not found: {dataset_file}")
        return

    with open(dataset_file, encoding="utf-8") as file:
        dataset = json.load(file)

    db = SessionLocal()

    methods = {
        "Vector Search": [],
        "BM25": [],
        "Hybrid Search": [],
        "Hybrid + Reranking": [],
    }

    try:
        for index, item in enumerate(dataset, start=1):
            question = item["question"]
            expected_ids = set(item["expected_chunk_ids"])

            print(f"\nEvaluating {index}/{len(dataset)}")

            # A. Vector search
            vector_results = retrieve_similar_chunks(
                db, document_id, question, top_k=TOP_K
            )

            methods["Vector Search"].append(
                find_rank(vector_results, expected_ids)
            )

            # B. BM25 search
            bm25_results = retrieve_bm25_chunks(
                db, document_id, question, top_k=TOP_K
            )

            methods["BM25"].append(
                find_rank(bm25_results, expected_ids)
            )

            # C. Hybrid search
            hybrid_results = retrieve_hybrid_chunks(
                db=db,
                document_id=document_id,
                query=question,
                candidate_k=CANDIDATE_K,
                top_k=TOP_K,
            )

            methods["Hybrid Search"].append(
                find_rank(hybrid_results, expected_ids)
            )

            # D. Hybrid + Cross-Encoder
            hybrid_candidates = retrieve_hybrid_chunks(
                db=db,
                document_id=document_id,
                query=question,
                candidate_k=CANDIDATE_K,
                top_k=CANDIDATE_K,
            )

            candidate_chunks = [
                chunk for chunk, score in hybrid_candidates
            ]

            reranked_results = rerank_chunks(
                query=question,
                chunks=candidate_chunks,
                top_k=TOP_K,
            )

            methods["Hybrid + Reranking"].append(
                find_rank(reranked_results, expected_ids)
            )

        # Display comparison
        print("\n" + "=" * 65)
        print("RETRIEVAL PERFORMANCE COMPARISON")
        print("=" * 65)

        print(f"Document ID: {document_id}")
        print(f"Total questions: {len(dataset)}")

        print(
            f"\n{'Method':<25} {'Recall@5':>12} {'MRR@5':>12}"
        )

        print("-" * 52)

        for method, ranks in methods.items():
            recall, mrr = calculate_metrics(ranks)

            print(
                f"{method:<25} "
                f"{recall:>11.2%} "
                f"{mrr:>12.4f}"
            )

    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m eval.compare_retrieval <document_id>")
        sys.exit(1)

    evaluate(int(sys.argv[1]))
