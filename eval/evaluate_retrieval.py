
import json
from pathlib import Path

from app.db.session import SessionLocal
from app.services.retriever import retrieve_similar_chunks_with_scores


DATASET_PATH = Path(__file__).parent / "retrieval_dataset.json"

DOCUMENT_ID = 4
TOP_K = 5


def evaluate_retrieval():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    db = SessionLocal()

    total_hits = 0

    try:
        for item in dataset:
            results = retrieve_similar_chunks_with_scores(
                db=db,
                document_id=DOCUMENT_ID,
                query=item["question"],
                top_k=TOP_K,
            )

            retrieved_pages = list(
                dict.fromkeys(
                    chunk.page_number for chunk, score in results
                )
            )

            expected_pages = item["expected_pages"]

            hit = any(
                page in retrieved_pages
                for page in expected_pages
            )

            if hit:
                total_hits += 1

            print("\n" + "=" * 60)
            print(f"Question: {item['question']}")
            print(f"Expected pages: {expected_pages}")
            print(f"Retrieved pages: {retrieved_pages}")
            print(f"Result: {'PASS' if hit else 'FAIL'}")

    finally:
        db.close()

    recall_at_5 = total_hits / len(dataset)

    print("\n" + "=" * 60)
    print(f"Total questions: {len(dataset)}")
    print(f"Successful retrievals: {total_hits}")
    print(f"Recall@5: {recall_at_5:.2%}")


if __name__ == "__main__":
    evaluate_retrieval()
