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


text = Path(
    "documents/knowledge.txt"
).read_text(encoding="utf-8")

chunks = chunk_by_paragraph(text)
chunk_embeddings = create_embeddings(chunks)
index = build_index(chunk_embeddings)


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
        top_k=3
    )

    retrieved_chunks = [
        chunks[int(i)]
        for i in indices
        if i >= 0
    ]

    context = "\n\n".join(retrieved_chunks)

    answer = generate_answer(
        question,
        context
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


