
import os
import uuid

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import Document, DocumentPage
from app.services.pdf_parser import extract_text
from app.services.chunker import create_document_chunks
from app.services.vector_store import generate_and_store_embeddings

router = APIRouter()

UPLOAD_DIR = "storage/pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # Validate PDF extension
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed",
        )

    unique_filename = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        # 1. Save uploaded PDF
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)

        # 2. Extract text from PDF
        pages = extract_text(file_path)

        if not pages:
            raise HTTPException(
                status_code=400,
                detail="No readable text found in PDF",
            )

        # 3. Create document record
        document = Document(
            filename=file.filename,
            file_path=file_path,
            status="processed",
        )

        db.add(document)
        db.flush()  # Generate document ID

        # 4. Save extracted page text
        for page in pages:
            db.add(
                DocumentPage(
                    document_id=document.id,
                    page_number=page["page_number"],
                    content=page["text"],
                )
            )

        # IMPORTANT: Flush page records before querying them
        db.flush()

        # 5. Generate and save chunks
        total_chunks = create_document_chunks(
            db=db,
            document_id=document.id,
        )
        # 6. Generate and save embeddings
        total_embeddings = generate_and_store_embeddings(
            db=db,
            document_id=document.id,
        )

        # 6. Commit everything together
        db.commit()
        db.refresh(document)

        return {
            "total_embeddings_created": total_embeddings,
            "message": "PDF uploaded, processed, chunked and embedded successfully",
            "document_id": document.id,
            "filename": document.filename,
            "total_pages_extracted": len(pages),
            "total_chunks_created": total_chunks,
            "status": document.status,
        }

    except HTTPException:
        db.rollback()

        if os.path.exists(file_path):
            os.remove(file_path)

        raise

    except Exception as e:
        db.rollback()

        if os.path.exists(file_path):
            os.remove(file_path)

        raise HTTPException(
            status_code=500,
            detail="PDF upload failed",
        ) from e

    finally:
        await file.close()
