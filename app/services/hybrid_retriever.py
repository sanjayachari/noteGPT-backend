
from sqlalchemy.orm import Session

from app.services.retriever import retrieve_similar_chunks
from app.services.bm25_retriever import retrieve_bm25_chunks


def reciprocal_rank_fusion(
    vector_results: list,
    bm25_results: list,
    top_k: int = 20,
    rrf_k: int = 60,
) -> list:

    # Store combined scores using chunk IDs
    fused_scores = {}
    chunk_lookup = {}

    # Process vector search rankings
    for rank, chunk in enumerate(vector_results, start=1):
        chunk_id = chunk.id

        fused_scores[chunk_id] = (
            fused_scores.get(chunk_id, 0)
            + 1 / (rrf_k + rank)
        )

        chunk_lookup[chunk_id] = chunk

    # Process BM25 rankings
    for rank, (chunk, score) in enumerate(bm25_results, start=1):
        chunk_id = chunk.id

        fused_scores[chunk_id] = (
            fused_scores.get(chunk_id, 0)
            + 1 / (rrf_k + rank)
        )

        chunk_lookup[chunk_id] = chunk

    # Sort by combined RRF score
    ranked_chunks = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        (chunk_lookup[chunk_id], score)
        for chunk_id, score in ranked_chunks[:top_k]
    ]


def retrieve_hybrid_chunks(
    db: Session,
    document_id: int,
    query: str,
    candidate_k: int = 20,
    top_k: int = 20,
) -> list:

    # 1. Vector search
    vector_results = retrieve_similar_chunks(
        db=db,
        document_id=document_id,
        query=query,
        top_k=candidate_k,
    )

    # 2. BM25 keyword search
    bm25_results = retrieve_bm25_chunks(
        db=db,
        document_id=document_id,
        query=query,
        top_k=candidate_k,
    )

    # 3. Combine both rankings using RRF
    return reciprocal_rank_fusion(
        vector_results=vector_results,
        bm25_results=bm25_results,
        top_k=top_k,
    )
