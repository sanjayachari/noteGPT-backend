
import tiktoken


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def split_text(self, text: str) -> list[str]:
        tokens = self.tokenizer.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk = self.tokenizer.decode(tokens[start:end])

            if chunk.strip():
                chunks.append(chunk)

            start += self.chunk_size - self.chunk_overlap

        return chunks


chunker = TextChunker()



from sqlalchemy.orm import Session

from app.db.models import DocumentPage, DocumentChunk


def create_document_chunks(
    db: Session,
    document_id: int,
) -> int:
    pages = (
        db.query(DocumentPage)
        .filter(DocumentPage.document_id == document_id)
        .order_by(DocumentPage.page_number)
        .all()
    )

    total_chunks = 0

    for page in pages:
        chunks = chunker.split_text(page.content)

        for index, content in enumerate(chunks):
            db.add(
                DocumentChunk(
                    document_id=document_id,
                    page_number=page.page_number,
                    chunk_index=index,
                    content=content,
                )
            )

            total_chunks += 1

    return total_chunks
