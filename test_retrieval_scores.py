
from app.db.session import SessionLocal
from app.services.retriever import retrieve_similar_chunks_with_scores

db = SessionLocal()

try:
    results = retrieve_similar_chunks_with_scores(
        db=db,
        document_id=4,
        query="What is the main objective of this research?",
        top_k=5,
    )

    for chunk, similarity in results:
        print("\n" + "=" * 60)
        print(f"Page: {chunk.page_number}")
        print(f"Chunk ID: {chunk.id}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Content: {chunk.content[:300]}")

finally:
    db.close()
