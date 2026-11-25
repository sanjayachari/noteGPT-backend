
from sentence_transformers import CrossEncoder

_model = None


def get_reranker():
    global _model

    if _model is None:
        _model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu",
        )

    return _model


def rerank_chunks(
    query: str,
    chunks: list,
    top_k: int = 5,
) -> list:

    model = get_reranker()

    pairs = [
        (query, chunk.content)
        for chunk in chunks
    ]

    scores = model.predict(pairs)

    ranked_results = sorted(
        zip(chunks, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return ranked_results[:top_k]
