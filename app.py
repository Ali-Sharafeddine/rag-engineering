from pathlib import Path
from fastapi import FastAPI, HTTPException
from schemas import QuestionRequest, QuestionResponse
from rag.chunking import chunk_by_paragraph
from rag.embeddings import create_embeddings
from rag.vector_store import build_index, search_index
from rag.generation import generate_answer


app = FastAPI(
    title="RAG Engineering API",
    version="1.0.0"
)


from config import KNOWLEDGE_PATH, TOP_K
text = KNOWLEDGE_PATH.read_text(
    encoding="utf-8"
)

chunks = chunk_by_paragraph(text)
chunk_embeddings = create_embeddings(chunks)
index = build_index(chunk_embeddings)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "rag-api"
    }


@app.post(
    "/ask",
    response_model=QuestionResponse
)
def ask_question(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    query_embedding = create_embeddings(
        [question]
    )[0]

    scores, indices = search_index(
        query_embedding,
        index,
        top_k=TOP_K
    )

    retrieved_chunks = [
        chunks[int(i)]
        for i in indices
        if i >= 0
    ]

    context = "\n\n".join(retrieved_chunks)

    try:
        answer = generate_answer(question, context)
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="The language model service is currently unavailable."
        )

    sources = []

    for score, index_position in zip(
        scores,
        indices
    ):
        if index_position >= 0:
            sources.append(
                {
                    "text": chunks[int(index_position)],
                    "score": float(score)
                }
            )

    return {
        "answer": answer,
        "sources": sources
    }


