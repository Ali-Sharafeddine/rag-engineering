from config import KNOWLEDGE_PATH
from rag.chunking import create_chunk_records
from rag.embeddings import create_embeddings
from rag.vector_store import build_index, search_index

evaluation_questions = [
    {
        "question": "What is Docker used for?",
        "expected_keyword": "Docker"
    },
    {
        "question": "What is RAG?",
        "expected_keyword": "Retrieval-Augmented Generation"
    },
    {
        "question": "What is SQLite?",
        "expected_keyword": "SQLite"
    },
    {
        "question": "What is Pydantic used for?",
        "expected_keyword": "Pydantic"
    },
    {
        "question": "Why is automated testing useful?",
        "expected_keyword": "testing"
    }
]

def evaluate_retrieval(
    evaluation_questions,
    chunks,
    index
):
    hit_at_1 = 0
    hit_at_3 = 0
    reciprocal_ranks = []

    for item in evaluation_questions:
        question = item["question"]
        expected_keyword = item["expected_keyword"]

        query_embedding = create_embeddings([question])[0]

        scores, indices = search_index(
            query_embedding,
            index,
            top_k=3
        )

        rank_found = None

        for rank, index_position in enumerate(
            indices,
            start=1
        ):
            chunk = chunks[int(index_position)]

            if expected_keyword.lower() in chunk.lower():
                rank_found = rank
                break

        if rank_found == 1:
            hit_at_1 += 1

        if rank_found is not None:
            hit_at_3 += 1
            reciprocal_ranks.append(
                1 / rank_found
            )
        else:
            reciprocal_ranks.append(0)

        print("Question:", question)
        print("Correct result rank:", rank_found)
        print()

    total = len(evaluation_questions)

    hit1_score = hit_at_1 / total
    hit3_score = hit_at_3 / total
    mrr = sum(reciprocal_ranks) / total

    print(f"Hit@1: {hit1_score:.2%}")
    print(f"Hit@3: {hit3_score:.2%}")
    print(f"MRR: {mrr:.3f}")


if __name__ == "__main__":
    text = KNOWLEDGE_PATH.read_text(
        encoding="utf-8"
    )

    chunk_records = create_chunk_records(
        text,
        source="knowledge.txt"
    )

    chunks = [
        record["text"]
        for record in chunk_records
    ]

    chunk_embeddings = create_embeddings(chunks)

    index = build_index(chunk_embeddings)

    evaluate_retrieval(
        evaluation_questions,
        chunks,
        index
    )