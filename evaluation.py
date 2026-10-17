from rag.embeddings import create_embeddings
from rag.vector_store import search_index

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
    correct = 0

    for item in evaluation_questions:
        question = item["question"]
        expected_keyword = item["expected_keyword"]

        query_embedding = create_embeddings([question])[0]

        scores, indices = search_index(
            query_embedding,
            index,
            top_k=1
        )

        top_chunk = chunks[indices[0]]

        print("Question:", question)
        print("Expected:", expected_keyword)
        print("Top result:", top_chunk)

        if expected_keyword.lower() in top_chunk.lower():
            print("PASS")
            correct += 1
        else:
            print("FAIL")

        print()

    total = len(evaluation_questions)
    accuracy = correct / total

    print(f"Retrieval Accuracy: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2%}")