
from app.services.groq_llm import get_groq_client, MODEL_NAME


def compress_context(
    question: str,
    context: str,
) -> str:

    client = get_groq_client()

    prompt = f"""
You are a context compression assistant for a RAG system.

Your task is to extract only the passages relevant
to answering the user's question.

Rules:
1. Preserve the original wording of relevant passages.
2. Do not invent or modify facts.
3. Do not generate an answer.
4. Remove irrelevant information.
5. Preserve document filenames exactly.
6. Preserve page numbers exactly.
7. Preserve all source labels and metadata.
8. Keep the original source-label format.
9. If no relevant information exists, return an empty response.

QUESTION:
{question}

DOCUMENT CONTEXT:
{context}

RELEVANT COMPRESSED CONTEXT:
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract relevant passages from documents "
                    "while preserving their original source metadata."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content.strip()
