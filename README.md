# AI PDF Assistant

An AI-powered PDF question-answering system built with **FastAPI, Python, PostgreSQL, pgvector and Large Language Models (LLMs)**.

The application allows users to upload PDF documents and ask questions about their contents. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information and generate context-grounded answers.

The project focuses on practical AI engineering concepts, including advanced retrieval strategies, semantic search, reranking, context compression and RAG evaluation.

## Key Features

### Document Processing
- PDF document parsing and text extraction.
- Document chunking with configurable strategies.
- Embedding generation using Sentence Transformers.
- Persistent document and embedding storage.
- PostgreSQL-based document management.

### Advanced RAG Pipeline
- Semantic search using vector embeddings.
- Keyword-based retrieval using BM25.
- Hybrid retrieval combining semantic and keyword search.
- Cross-encoder reranking for improved relevance.
- Multi-query retrieval to improve search coverage.
- Query rewriting for better retrieval.
- Context compression to reduce unnecessary LLM context.
- Context-grounded answer generation.

### Backend Engineering
- FastAPI REST API architecture.
- Modular service-based code organization.
- PostgreSQL database integration.
- Database migrations using Alembic.
- Authentication and security utilities.
- Structured request and response schemas.
- Application logging and error handling.

### RAG Evaluation
- Retrieval evaluation datasets.
- Comparison of retrieval strategies.
- Reranking performance comparisons.
- Dynamic evaluation workflows.
- Retrieval quality analysis.

## Architecture

```text
                  User
                   |
                   v
              FastAPI API
                   |
          +--------+--------+
          |                 |
          v                 v
    Document Upload     Chat Endpoint
          |                 |
          v                 v
      PDF Parser       Query Rewriter
          |                 |
          v                 v
       Chunking       Multi-Query Search
          |                 |
          v                 v
     Embeddings       Hybrid Retrieval
          |                 |
          v                 v
      PostgreSQL      Candidate Results
      + pgvector            |
                            v
                       Reranking
                            |
                            v
                    Context Compression
                            |
                            v
                       LLM Generation
                            |
                            v
                      Final Answer
```

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Backend | FastAPI |
| Database | PostgreSQL |
| Vector Search | pgvector |
| Embeddings | Sentence Transformers |
| Retrieval | BM25, Semantic Search |
| Reranking | Cross-Encoder |
| AI / LLM | Gemini / configured LLM provider |
| Database Migrations | Alembic |
| Validation | Pydantic |
| Dependency Management | uv |
| Testing & Evaluation | Python, custom evaluation scripts |

## Project Structure

```text
backend/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── chat.py
│   │       ├── documents.py
│   │       └── health.py
│   ├── core/
│   │   ├── logging.py
│   │   └── security.py
│   ├── db/
│   │   ├── models.py
│   │   └── session.py
│   ├── schemas/
│   ├── services/
│   │   ├── bm25_retriever.py
│   │   ├── chunker.py
│   │   ├── context_compressor.py
│   │   ├── embeddings.py
│   │   ├── hybrid_retriever.py
│   │   ├── llm.py
│   │   ├── multi_query.py
│   │   ├── pdf_parser.py
│   │   ├── query_rewriter.py
│   │   ├── reranker.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   └── main.py
├── eval/
│   ├── datasets/
│   ├── compare_reranking.py
│   ├── compare_retrieval.py
│   ├── evaluate_dynamic.py
│   ├── evaluate_retrieval.py
│   └── generate_dataset.py
├── migrations/
├── storage/
├── tests/
├── alembic.ini
├── pyproject.toml
├── uv.lock
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.12+
- PostgreSQL
- pgvector extension
- uv package manager
- API key for your configured LLM provider

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ai-pdf-assistant/backend
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment variables

Create a `.env` file in the backend root.

Configure your database connection, LLM API credentials and other application settings according to `app/config.py`.

Never commit your `.env` file or API credentials.

### 4. Configure PostgreSQL

Create a PostgreSQL database and enable pgvector:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 5. Run database migrations

```bash
uv run alembic upgrade head
```

### 6. Start the application

```bash
uv run uvicorn app.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

## AI Engineering Concepts Demonstrated

This project explores several important concepts used in production RAG systems:

1. **Semantic Retrieval:** Finding relevant information based on meaning rather than exact keyword matching.
2. **Hybrid Search:** Combining BM25 keyword retrieval with vector similarity search.
3. **Reranking:** Using a Cross-Encoder to improve the relevance ordering of retrieved candidates.
4. **Query Rewriting:** Transforming user questions into better search queries.
5. **Multi-Query Retrieval:** Generating multiple search variations to improve document recall.
6. **Context Compression:** Reducing irrelevant information before sending context to an LLM.
7. **RAG Evaluation:** Measuring and comparing retrieval quality across different strategies.

## Evaluation

The repository includes evaluation scripts to experiment with and compare retrieval approaches.

The evaluation workflow covers:

- Retrieval strategy comparisons.
- Hybrid search experiments.
- Reranking effectiveness.
- Evaluation dataset generation.
- Retrieval quality analysis.

## Future Improvements

- Advanced RAG evaluation using RAGAS or DeepEval.
- Streaming responses.
- Improved document ingestion and parsing.
- Advanced agentic workflows.
- LLM observability and tracing.
- Production deployment and scalability improvements.
- Comprehensive automated test coverage.

## Project Goal

The goal of this project is to understand and implement the engineering behind production-ready AI applications rather than simply building a basic chatbot around an LLM API.

It serves as a practical exploration of modern AI engineering, RAG architecture, information retrieval and backend system design.

---

**Author:** Sanjay Achari

**GitHub:** https://github.com/sanjayachari

**Portfolio:** https://sanjay-developer-portfolio.vercel.app/