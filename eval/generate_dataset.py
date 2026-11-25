
import json
import sys
from pathlib import Path

from app.db.session import SessionLocal
from app.db.models import Document, DocumentChunk
from app.services.llm import generate_evaluation_questions


OUTPUT_DIR = Path(__file__).parent / "datasets"


def generate_dataset(document_id: int):
    db = SessionLocal()

    try:
        # 1. Check whether the document exists
        document = (
            db.query(Document)
            .filter(Document.id == document_id)
            .first()
        )

        if not document:
            print(f"Document {document_id} not found.")
            return

        # 2. Fetch all chunks belonging to this PDF
        chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )

        if not chunks:
            print("No chunks found for this document.")
            return

        print(f"\nGenerating questions for: {document.filename}")
        print(f"Total chunks: {len(chunks)}")

        dataset = []

        # 3. Generate questions from each chunk
        for index, chunk in enumerate(chunks, start=1):
            print(
                f"\nProcessing chunk {index}/{len(chunks)} "
                f"(Page {chunk.page_number})..."
            )

            questions = generate_evaluation_questions(
                context=chunk.content
            )

            # 4. Attach original source information
            for item in questions:
                dataset.append({
                    "document_id": document_id,
                    "question": item["question"],
                    "expected_answer": item["expected_answer"],
                    "expected_chunk_ids": [chunk.id],
                    "expected_pages": [chunk.page_number],
                })

        # 5. Save the generated dataset
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        output_file = OUTPUT_DIR / f"document_{document_id}.json"

        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(dataset, file, indent=2, ensure_ascii=False)

        print("\n" + "=" * 50)
        print("DATASET GENERATION COMPLETED")
        print(f"Document: {document.filename}")
        print(f"Questions generated: {len(dataset)}")
        print(f"Saved to: {output_file}")

    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m eval.generate_dataset <document_id>")
        sys.exit(1)

    document_id = int(sys.argv[1])

    generate_dataset(document_id)
