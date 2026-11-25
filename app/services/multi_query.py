
import json
import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def generate_multiple_queries(question: str) -> list[str]:
    prompt = f"""
You are a query generation assistant for a PDF search system.

Generate 3 different search queries based on the user's question.

Rules:
- Preserve the original meaning.
- Use different wording for better document retrieval.
- Do not answer the question.
- Do not introduce unrelated information.
- Return valid JSON only.

USER QUESTION:
{question}

Return exactly this JSON structure:
{{
    "queries": [
        "query 1",
        "query 2",
        "query 3"
    ]
}}
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.3
        }
    }

    response = httpx.post(
        OLLAMA_URL,
        json=payload,
        timeout=120.0,
    )

    response.raise_for_status()

    result = json.loads(response.json()["response"])

    return result["queries"]
