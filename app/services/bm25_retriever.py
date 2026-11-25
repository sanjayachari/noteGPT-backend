
import re

from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session

from app.db.models import DocumentChunk


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def retrieve_bm25_chunks(
    db: Session,
    document_id: int,
    query: str,
    top_k: int = 20,
) -> list[tuple[DocumentChunk, float]]:

    # 1. Fetch chunks belonging to this PDF
    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .all()
    )

    if not chunks:
        return []

    # 2. Tokenize every chunk
    tokenized_chunks = [
        tokenize(chunk.content)
        for chunk in chunks
    ]

    # 3. Build BM25 index
    bm25 = BM25Okapi(tokenized_chunks)

    # 4. Tokenize the user's question
    tokenized_query = tokenize(query)

    if not tokenized_query:
        return []

    # 5. Calculate BM25 relevance scores
    scores = bm25.get_scores(tokenized_query)

    # 6. Sort chunks by relevance
    ranked_results = sorted(
        zip(chunks, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return [
        (chunk, float(score))
        for chunk, score in ranked_results[:top_k]
    ]
