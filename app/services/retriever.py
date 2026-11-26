
from sqlalchemy.orm import Session

from app.db.models import DocumentChunk
from app.services.embeddings import generate_embedding


def retrieve_similar_chunks(
    db: Session,
    document_id: int | None,
    query: str,
    top_k: int = 5,
    page_number: int | None = None,
    document_ids: list[int] | None = None,
) -> list[DocumentChunk]:

    # 1. Convert the user's question into an embedding
    query_embedding = generate_embedding(query)

    # 2. Build the retrieval query
    query_builder = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.embedding.is_not(None))
    )

    # 3. Filter by multiple documents or a single document
    if document_ids is not None:
        query_builder = query_builder.filter(
            DocumentChunk.document_id.in_(document_ids)
        )
    elif document_id is not None:
        query_builder = query_builder.filter(
            DocumentChunk.document_id == document_id
        )
    else:
        raise ValueError(
            "Either document_id or document_ids must be provided"
        )

    # 4. Apply page filter when provided
    if page_number is not None:
        query_builder = query_builder.filter(
            DocumentChunk.page_number == page_number
        )

    # 5. Retrieve the closest chunks
    chunks = (
        query_builder
        .order_by(
            DocumentChunk.embedding.cosine_distance(query_embedding)
        )
        .limit(top_k)
        .all()
    )

    return chunks


def retrieve_similar_chunks_with_scores(
    db: Session,
    document_id: int | None,
    query: str,
    top_k: int = 5,
    page_number: int | None = None,
    document_ids: list[int] | None = None,
) -> list:

    # 1. Generate query embedding
    query_embedding = generate_embedding(query)

    # 2. Calculate cosine distance
    distance = DocumentChunk.embedding.cosine_distance(query_embedding)

    # 3. Build retrieval query
    query_builder = (
        db.query(
            DocumentChunk,
            (1 - distance).label("similarity"),
        )
        .filter(DocumentChunk.embedding.is_not(None))
    )

    # 4. Filter by multiple documents or a single document
    if document_ids is not None:
        query_builder = query_builder.filter(
            DocumentChunk.document_id.in_(document_ids)
        )
    elif document_id is not None:
        query_builder = query_builder.filter(
            DocumentChunk.document_id == document_id
        )
    else:
        raise ValueError(
            "Either document_id or document_ids must be provided"
        )

    # 5. Apply page filter
    if page_number is not None:
        query_builder = query_builder.filter(
            DocumentChunk.page_number == page_number
        )

    # 6. Retrieve results with similarity scores
    results = (
        query_builder
        .order_by(distance)
        .limit(top_k)
        .all()
    )

    return results
