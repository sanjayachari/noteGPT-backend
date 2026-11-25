from sqlalchemy.orm import Session

from app.db.models import DocumentChunk
from app.services.embeddings import generate_embedding


def generate_and_store_embeddings(
    db: Session,
    document_id: int,
) -> int:
    """Generate and store embeddings for chunks that don't have one yet."""

    # Ensure newly added chunks are visible to the query
    db.flush()

    chunks = (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.document_id == document_id,
            DocumentChunk.embedding.is_(None),
        )
        .all()
    )

    for chunk in chunks:
        chunk.embedding = generate_embedding(chunk.content)

    db.commit()

    return len(chunks)