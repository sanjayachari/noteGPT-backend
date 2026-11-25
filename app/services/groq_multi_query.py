
import json

from app.services.groq_llm import get_groq_client, MODEL_NAME


def generate_multiple_queries(question: str) -> list[str]:

    client = get_groq_client()

    prompt = f"""
You are a search query generation assistant.

Generate exactly 3 alternative search queries for the
given question.

Each query should explore the same information need
using different wording or relevant terminology.

Rules:
1. Preserve the original intent.
2. Use different wording for each query.
3. Do not answer the question.
4. Avoid unrelated queries.
5. Return valid JSON only.

Required JSON format:

{{
    "queries": [
        "First alternative query",
        "Second alternative query",
        "Third alternative query"
    ]
}}

QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You generate alternative search queries.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.3,
        max_tokens=512,
        response_format={"type": "json_object"},
    )

    content = (response.choices[0].message.content or "").strip()
    data = json.loads(content) if content else {}

    queries = data.get("queries", [])

    return [
        query.strip()
        for query in queries
        if isinstance(query, str) and query.strip()
    ][:3]
