
from app.services.groq_llm import get_groq_client, MODEL_NAME


def rewrite_query(
    question: str,
    chat_history: str = "",
) -> str:

    client = get_groq_client()

    prompt = f"""
You are a search query rewriting assistant.

Your task is to rewrite the user's latest question into
a standalone search query that can be used to retrieve
relevant information from documents.

Rules:
1. Resolve references using the conversation history.
2. Preserve the original meaning.
3. Do not answer the question.
4. Do not add unrelated information.
5. Return only the rewritten question.

CONVERSATION HISTORY:
{chat_history}

LATEST QUESTION:
{question}

REWRITTEN QUERY:
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You rewrite questions for document retrieval.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.1,
        max_tokens=512,
    )

    return (response.choices[0].message.content or "").strip()
