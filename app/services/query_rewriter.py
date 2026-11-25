
import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def rewrite_query(
    question: str,
    chat_history: str = "",
) -> str:

    prompt = f"""
You are a query rewriting assistant for a PDF search system.

Your task is to convert the user's question into a clear,
standalone search query.

Rules:
- Preserve the original meaning.
- Resolve references using conversation history.
- Do not answer the question.
- Do not add information that the user did not provide.
- If the question is already clear, keep it unchanged.
- Return only the rewritten search query.

CONVERSATION HISTORY:
{chat_history}

USER QUESTION:
{question}

REWRITTEN QUERY:
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    response = httpx.post(
        OLLAMA_URL,
        json=payload,
        timeout=120.0
    )

    response.raise_for_status()

    return response.json()["response"].strip()
