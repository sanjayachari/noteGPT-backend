
import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from google.genai.errors import ServerError

from app.db.session import get_db
from app.db.models import Document, Conversation, ChatMessage

from app.services.retriever import retrieve_similar_chunks

# Existing Ollama imports retained for reference
# from app.services.llm import generate_answer
# from app.services.query_rewriter import rewrite_query
# from app.services.multi_query import generate_multiple_queries
# from app.services.context_compressor import compress_context

from app.services.groq_llm import generate_answer
from app.services.groq_query_rewriter import rewrite_query
from app.services.groq_multi_query import generate_multiple_queries
from app.services.groq_context_compressor import compress_context

router = APIRouter()


# Handle Gemini API errors
def call_gemini(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)

    except ServerError as error:
        if getattr(error, "code", None) == 503:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "AI_SERVICE_BUSY",
                    "message": (
                        "Our AI service is currently experiencing "
                        "high demand. Please try again in a few moments."
                    ),
                    "retryable": True,
                },
                headers={"Retry-After": "30"},
            ) from error

        raise


class ChatRequest(BaseModel):
    document_id: int
    document_ids: list[int] | None = None
    question: str
    conversation_id: int | None = None
    page_number: int | None = Field(default=None, ge=1)


def match_passage_to_chunk(passage: str, chunks: list) -> tuple | None:
    """
    Match an extracted passage against the original chunks.

    The compressor's page label is ignored.
    Original database metadata is authoritative.
    """
    passage_tokens = set(re.findall(r"\w+", passage.lower()))

    if len(passage_tokens) < 5:
        return None

    best_chunk = None
    best_score = 0.0

    for chunk in chunks:
        chunk_tokens = set(
            re.findall(r"\w+", chunk.content.lower())
        )

        if not chunk_tokens:
            continue

        overlap = len(passage_tokens & chunk_tokens)
        score = overlap / len(passage_tokens)

        if score > best_score:
            best_score = score
            best_chunk = chunk

    if best_score >= 0.60:
        return best_chunk

    return None


def recover_compressed_passages(
    compressed_context: str,
    chunks: list,
    documents_by_id: dict,
) -> tuple[str, list]:
    """
    Recover correct document and page labels for compressed passages.

    Returns:
        rebuilt_context: passages with verified source labels
        matched_chunks: original database chunks matched to passages
    """
    pattern = re.compile(
        r"Relevant extracted passage:\s*(.*?)"
        r"(?=\n\s*\[Document:|\Z)",
        re.DOTALL,
    )

    matches = pattern.findall(compressed_context)

    rebuilt_parts = []
    matched_chunks = {}

    for passage in matches:
        passage = re.sub(
            r"(?im)^\s*No relevant context found\.?\s*$",
            "",
            passage,
        ).strip()

        if not passage:
            continue

        chunk = match_passage_to_chunk(passage, chunks)

        if chunk is None:
            continue

        document = documents_by_id[chunk.document_id]

        rebuilt_parts.append(
            f"[Document: {document.filename} | Page {chunk.page_number}]\n"
            f"[Page {chunk.page_number}]\n"
            f"{passage}"
        )

        matched_chunks[chunk.id] = chunk

    rebuilt_context = "\n\n".join(rebuilt_parts)

    return rebuilt_context, list(matched_chunks.values())


@router.post("/chat")
def chat_with_pdf(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    # 1. Determine selected documents
    selected_document_ids = list(dict.fromkeys(
        request.document_ids or [request.document_id]
    ))

    if request.document_id not in selected_document_ids:
        raise HTTPException(
            status_code=400,
            detail="document_id must be included in document_ids",
        )

    # 2. Validate selected documents
    documents = (
        db.query(Document)
        .filter(Document.id.in_(selected_document_ids))
        .all()
    )

    documents_by_id = {
        document.id: document for document in documents
    }

    missing_ids = [
        document_id
        for document_id in selected_document_ids
        if document_id not in documents_by_id
    ]

    if missing_ids:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "One or more documents were not found",
                "missing_document_ids": missing_ids,
            },
        )

    primary_document = documents_by_id[request.document_id]

    # 3. Detect filenames explicitly mentioned in the question
    question_lower = request.question.lower()

    mentioned_documents = [
        document
        for document in documents
        if document.filename.lower() in question_lower
    ]

    if mentioned_documents:
        retrieval_document_ids = [
            document.id for document in mentioned_documents
        ]
    else:
        retrieval_document_ids = selected_document_ids

    # 4. Get an existing conversation or create a new one
    if request.conversation_id is not None:
        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.id == request.conversation_id,
                Conversation.document_id == request.document_id,
            )
            .first()
        )

        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found for this document",
            )
    else:
        conversation = Conversation(
            document_id=primary_document.id,
            title=request.question[:255],
        )

        db.add(conversation)
        db.flush()

    # 5. Load previous conversation messages
    previous_messages = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.conversation_id == conversation.id
        )
        .order_by(
            ChatMessage.created_at.asc(),
            ChatMessage.id.asc(),
        )
        .all()
    )

    recent_messages = previous_messages[-10:]

    chat_history = "\n".join(
        f"{message.role}: {message.content}"
        for message in recent_messages
    )

    # 6. Rewrite the question
    rewritten_query = call_gemini(
        rewrite_query,
        question=request.question,
        chat_history=chat_history,
    )

    # 7. Generate alternative queries
    alternative_queries = call_gemini(
        generate_multiple_queries,
        rewritten_query,
    )

    all_queries = list(dict.fromkeys(
        [rewritten_query] + alternative_queries
    ))

    # 8. Retrieve top 5 chunks separately from each document
    retrieved_chunks = {}

    for query in all_queries:
        for document_id in retrieval_document_ids:
            query_chunks = retrieve_similar_chunks(
                db=db,
                document_id=document_id,
                query=query,
                top_k=5,
                page_number=request.page_number,
            )

            for chunk in query_chunks:
                retrieved_chunks[chunk.id] = chunk

    chunks = list(retrieved_chunks.values())

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant content found in the selected documents",
        )

    # 9. Build original context using database metadata
    context_parts = []

    for chunk in chunks:
        document = documents_by_id[chunk.document_id]

        context_parts.append(
            f"[Document: {document.filename} | Page {chunk.page_number}]\n"
            f"[Page {chunk.page_number}]\n"
            f"{chunk.content}"
        )

    context = "\n\n".join(context_parts)

    # 10. Compress retrieved context
    compressed_context = call_gemini(
        compress_context,
        question=request.question,
        context=context,
    )

    # 11. Recover correct document/page labels
    answer_context, source_chunks = recover_compressed_passages(
        compressed_context=compressed_context,
        chunks=chunks,
        documents_by_id=documents_by_id,
    )

    no_context = (
        compressed_context.strip().lower()
        == "no relevant context found."
    )

    if no_context:
        answer_context = ""
        source_chunks = []

    elif not answer_context:
        # Compression returned an unexpected format.
        # Use original context and cite its retrieved chunks.
        answer_context = context
        source_chunks = chunks

    # 12. Generate the answer
    if not answer_context:
        answer = "I couldn't find this information in the uploaded PDF."
    else:
        source_pages = sorted(set(
            chunk.page_number for chunk in source_chunks
        ))

        answer = call_gemini(
            generate_answer,
            question=request.question,
            context=answer_context,
            source_pages=source_pages,
        )

    # 13. Build deduplicated sources
    source_map = {}

    for chunk in source_chunks:
        key = (chunk.document_id, chunk.page_number)

        if key not in source_map:
            source_map[key] = {
                "document_id": chunk.document_id,
                "filename": documents_by_id[chunk.document_id].filename,
                "page_number": chunk.page_number,
                "chunk_id": chunk.id,
            }

    sources = list(source_map.values())

    # 14. Append deterministic source references
    if sources:
        source_lines = [
            f"- [{source['filename']}, Page {source['page_number']}]"
            for source in sources
        ]

        answer = (
            answer.rstrip()
            + "\n\nSources:\n"
            + "\n".join(source_lines)
        )

    # 15. Save user message
    user_message = ChatMessage(
        conversation_id=conversation.id,
        role="user",
        content=request.question,
    )

    db.add(user_message)

    # 16. Save assistant answer
    assistant_message = ChatMessage(
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
    )

    db.add(assistant_message)

    # 17. Update conversation and commit
    conversation.updated_at = func.now()

    db.commit()

    # 18. Return response
    return {
        "document_id": request.document_id,
        "document_ids": selected_document_ids,
        "retrieval_document_ids": retrieval_document_ids,
        "conversation_id": conversation.id,
        "question": request.question,
        "rewritten_query": rewritten_query,
        "alternative_queries": alternative_queries,
        "page_filter": request.page_number,
        "original_context": context,
        "compressed_context": compressed_context,
        "answer_context": answer_context,
        "answer": answer,
        "sources": sources,
    }
