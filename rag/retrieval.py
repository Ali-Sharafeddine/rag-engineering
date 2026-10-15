import numpy as np

from rag.embeddings import create_embeddings


def cosine_similarity(vector_a, vector_b):
    numerator = np.dot(vector_a, vector_b)

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    return numerator / denominator

def retrieve(
    query: str,
    chunks: list[str],
    chunk_embeddings,
    top_k: int = 3
):
    query_embedding = create_embeddings([query])[0]

    scores = []

    for embedding in chunk_embeddings:
        score = cosine_similarity(
            query_embedding,
            embedding
        )

        scores.append(score)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append(
            (
                chunks[index],
                float(scores[index])
            )
        )

    return results