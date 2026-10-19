Built a Retrieval-Augmented Generation pipeline from scratch using paragraph-based chunking, SentenceTransformers embeddings, FAISS semantic search, and an LLM for grounded answer generation.


I first implemented cosine similarity manually, then replaced it with FAISS.

End-to-end RAG system with paragraph chunking, SentenceTransformer embeddings, FAISS semantic retrieval, grounded LLM generation, FastAPI serving, source attribution, retrieval evaluation, and automated API tests.

## Current Features

- Paragraph-based document chunking
- SentenceTransformer embeddings
- FAISS semantic retrieval
- Grounded LLM generation
- Source attribution with similarity scores
- Retrieval evaluation
- FastAPI `/ask` endpoint
- FastAPI `/health` endpoint
- Centralized configuration
- Automated API tests with mocked LLM generation
- Dockerfile for containerized deployment

## Test Status

- Retrieval evaluation: 5/5 expected top results
- API tests: 4 passed
