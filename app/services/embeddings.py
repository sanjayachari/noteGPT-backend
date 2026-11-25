from sentence_transformers import SentenceTransformer

# Load the model once when the application starts
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def generate_embedding(text: str) -> list[float]:
    """Generate a 384-dimensional embedding for the given text."""
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()