
from app.db.session import SessionLocal
from app.services.hybrid_retriever import retrieve_hybrid_chunks
from app.services.reranker import rerank_chunks


db = SessionLocal()

try:
    query = (
        "According to Holec (1981), "
        "what is the definition of autonomy in language learning?"
    )

    # 1. Retrieve hybrid candidates
    hybrid_results = retrieve_hybrid_chunks(
        db=db,
        document_id=6,
        query=query,
        candidate_k=20,
        top_k=20,
    )

    # 2. Extract chunk objects
    candidate_chunks = [
        chunk for chunk, score in hybrid_results
    ]

    # 3. Rerank using Cross-Encoder
    reranked_results = rerank_chunks(
        query=query,
        chunks=candidate_chunks,
        top_k=5,
    )

    print("\nHYBRID + CROSS-ENCODER RESULTS")
    print("=" * 60)

    for rank, (chunk, score) in enumerate(
        reranked_results,
        start=1,
    ):
        print(f"\nRank: {rank}")
        print(f"Chunk ID: {chunk.id}")
        print(f"Page: {chunk.page_number}")
        print(f"Cross-Encoder Score: {float(score):.4f}")
        print(f"Content: {chunk.content[:300]}")

finally:
    db.close()
