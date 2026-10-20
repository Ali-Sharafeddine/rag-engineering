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


from config import KNOWLEDGE_PATH, TOP_K, MIN_RETRIEVAL_SCORE
text = KNOWLEDGE_PATH.read_text(
    encoding="utf-8"
)

from rag.chunking import create_chunk_records

chunk_records = create_chunk_records(
    text,
    source="knowledge.txt"
)


chunk_texts = [
    record["text"]
    for record in chunk_records
]


chunk_embeddings = create_embeddings(chunk_texts)
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

    # 1. Create sources FIRST
    sources = []

    # 2. Fill sources with results above threshold
    for score, index_position in zip(scores, indices):
        if (
            index_position >= 0
            and float(score) >= MIN_RETRIEVAL_SCORE
        ):
            record = chunk_records[int(index_position)]

            sources.append(
                {
                    "text": record["text"],
                    "score": float(score),
                    "source": record["source"]
                }
            )

    # 3. If nothing useful was retrieved
    if not sources:
        return {
            "answer": (
                "I do not have enough relevant information "
                "to answer this question."
            ),
            "sources": []
        }

    # 4. Build context only AFTER sources exists
    context = "\n\n".join(
        source["text"]
        for source in sources
    )

    # 5. Generate answer
    try:
        answer = generate_answer(
            question,
            context
        )
    except Exception:
        raise HTTPException(
            status_code=503,
            detail=(
                "The language model service "
                "is currently unavailable."
            )
        )

    # 6. Return answer + filtered sources
    return {
        "answer": answer,
        "sources": sources
    }
    




