import faiss
import numpy as np


def build_index(embeddings):
    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index

def search_index(
    query_embedding,
    index,
    top_k: int = 3
):
    query_embedding = np.asarray(
        [query_embedding],
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    return scores[0], indices[0]