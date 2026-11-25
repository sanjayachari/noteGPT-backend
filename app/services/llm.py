
import json

import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def generate_answer(
    question: str,
    context: str,
    source_pages: list[int] | None = None,
) -> str:

    pages = sorted(set(source_pages or []))

    page_information = (
        ", ".join(str(page) for page in pages)
        if pages
        else "Not provided"
    )

    prompt = f"""
You are an accurate AI assistant answering questions
using information extracted from uploaded PDFs.

STRICT RULES:
- Answer using only the provided PDF context.
- Never invent facts or use outside knowledge.
- If the answer is unavailable, say:
  "I couldn't find this information in the uploaded PDF."
- Give a clear and direct answer.
- Include relevant details when the question asks for multiple points.
- Use bullet points or numbered lists when appropriate.
- Avoid repeating the same information.
- Do not generate citations, page references, filenames,
  or a Sources section. The application adds verified sources separately.
- Never include unsupported claims just to make the answer
  more complete.
- Never put quotation marks around paraphrased information.
- Only use direct quotations when the exact wording appears in the context.
- Clearly distinguish information directly stated in the PDF
  from reasonable interpretations.

AVAILABLE PDF PAGE NUMBERS:
{page_information}

PDF CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
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


def generate_evaluation_questions(context: str) -> list[dict]:
    prompt = f"""
You are creating evaluation questions for a PDF retrieval system.

Read the provided text and generate 2 questions that can be
answered directly using the text.

Rules:
- Questions must be clear and natural.
- Questions must test understanding of the provided text.
- Answers must be supported by the provided text.
- Do not use outside knowledge.
- Avoid vague questions.
- Do not invent information.

Return valid JSON using exactly this structure:

{{
    "questions": [
        {{
            "question": "Question here",
            "expected_answer": "Answer supported by the text"
        }}
    ]
}}

TEXT:
{context}
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
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

    result = json.loads(response.json()["response"])

    return result["questions"]
