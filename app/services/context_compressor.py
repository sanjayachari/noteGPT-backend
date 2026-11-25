
import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def compress_context(question: str, context: str) -> str:
    prompt = f"""
You are a precise information extraction system.

Extract the passages from the PDF context that help answer
the user's question.

IMPORTANT RULES:
- Extract relevant sentences and supporting facts.
- Preserve the original wording and meaning.
- Do not invent, rewrite, or answer the question.
- Keep relevant author names, dates and citations.
- Every extracted passage MUST include its original source label.
- Copy the complete source label exactly as it appears in the input.
- Never remove, change, or guess a document filename.
- Never remove, change, or guess a PDF page number.
- Do not combine passages from different pages or documents.
- If multiple passages come from the same page, you may group them
  under the same source label.
- Do not include passages without their source labels.
- If nothing is relevant, return exactly:
  No relevant context found.

REQUIRED OUTPUT FORMAT:

[Document: exact_filename.pdf | Page N]
Relevant extracted passage.

[Document: another_filename.pdf | Page N]
Another relevant extracted passage.

USER QUESTION:
{question}

PDF CONTEXT:
{context}

RELEVANT EXTRACTED CONTEXT:
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }

    timeout = httpx.Timeout(
        connect=10.0,
        read=600.0,
        write=30.0,
        pool=10.0,
    )

    response = httpx.post(
        OLLAMA_URL,
        json=payload,
        timeout=timeout,
    )

    response.raise_for_status()

    return response.json()["response"].strip()
        