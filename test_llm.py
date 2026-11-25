from app.services.llm import generate_answer

answer = generate_answer(
    question="What is RAG?",
    context=(
        "RAG stands for Retrieval-Augmented Generation. "
        "It retrieves relevant information from external documents "
        "and provides it to a language model to generate grounded answers."
    ),
)

print("\nGenerated Answer:")
print(answer)