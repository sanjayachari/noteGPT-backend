
import json
import os
from functools import lru_cache

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)

@lru_cache(maxsize=1)
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured")

    return Groq(api_key=api_key)

def generate_answer(
    question: str,
    context: str,
    source_pages: list | None = None,
) -> str:

    client = get_groq_client()

    source_instruction = ""

    if source_pages:
        source_instruction = (
            "\nUse the provided document and page information "
            "when referencing sources."
        )

    prompt = f"""
You are an AI assistant that answers questions using the
provided document context.

Rules:
1. Answer using only the provided context.
2. Do not invent information.
3. If the answer is not available in the context,
   clearly say that the documents do not contain the answer.
4. Give clear and concise answers.
5. Preserve important technical details.
6. Do not mention information that is not supported by the context.
{source_instruction}

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful document question-answering assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    return (response.choices[0].message.content or "").strip()


def generate_evaluation_questions(context: str) -> list[dict]:

    client = get_groq_client()

    prompt = f"""
Generate 2 meaningful question-answer pairs from the
following document context.

Return ONLY valid JSON in this format:

[
    {{
        "question": "Question here",
        "answer": "Answer here"
    }}
]

Do not include Markdown or explanations.

DOCUMENT CONTEXT:
{context}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content.strip()
    data = json.loads(content)

    if isinstance(data, dict):
        data = data.get("questions", [])

    return data
