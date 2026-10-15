from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def generate_answer(question: str, context: str) -> str:
    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=(
            "Answer the question using only the provided context. "
            "If the context does not contain the answer, "
            "say that you do not have enough information."
        ),
        input=f"Context:\n{context}\n\nQuestion:\n{question}"
    )

    return response.output_text