from openai import OpenAI

from config import LLM_MODEL


def generate_answer(
    question: str,
    context: str
) -> str:

    client = OpenAI()

    response = client.responses.create(
        model=LLM_MODEL,
        instructions=(
            "Answer the question using only the provided context. "
            "If the context does not contain the answer, "
            "say that you do not have enough information."
        ),
        input=(
            f"Context:\n{context}\n\n"
            f"Question:\n{question}"
        )
    )

    return response.output_text