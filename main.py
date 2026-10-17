from pathlib import Path
from rag.chunking import chunk_by_paragraph
from rag.embeddings import create_embeddings
from rag.vector_store import build_index, search_index
from rag.generation import generate_answer
from evaluation import evaluate_retrieval, evaluation_questions


text = Path(
    "documents/knowledge.txt"
).read_text(encoding="utf-8")

chunks = chunk_by_paragraph(text)

chunk_embeddings = create_embeddings(chunks)

index = build_index(chunk_embeddings)

query = "What is RAG?"

query_embedding = create_embeddings([query])[0]

scores, indices = search_index(
    query_embedding,
    index,
    top_k=3
)

for score, index_position in zip(scores, indices):
    print("\nScore:", float(score))
    print("Chunk:", chunks[index_position])


context = "\n\n".join(
    chunks[int(i)]
    for i in indices
    if i >= 0
)

answer = generate_answer(query, context)

print("\nQuestion:", query)
print("\nGenerated answer:", answer)

print("\nSources:")

for rank, (score, index_position) in enumerate(
    zip(scores, indices),
    start=1
):
    print(f"\nSource {rank}")
    print(f"Score: {float(score):.4f}")
    print(chunks[int(index_position)])

evaluate_retrieval(
    evaluation_questions,
    chunks,
    index
)
