from fastapi.testclient import TestClient
from config import MIN_RETRIEVAL_SCORE
from app import app

client = TestClient(app)


def test_ask_docker(monkeypatch):

    def fake_generate_answer(question, context):
        return "Docker runs applications inside isolated containers."

    monkeypatch.setattr(
        "app.generate_answer",
        fake_generate_answer
    )

    response = client.post(
        "/ask",
        json={
            "question": "What is Docker used for?"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "sources" in data

    assert (
        data["answer"]
        == "Docker runs applications inside isolated containers."
    )

    assert len(data["sources"]) >= 1
    assert "Docker" in data["sources"][0]["text"]


def test_empty_question():

    response = client.post(
        "/ask",
        json={
            "question": ""
        }
    )

    assert response.status_code == 422


def test_whitespace_question():

    response = client.post(
        "/ask",
        json={
            "question": "   "
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty."


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "rag-api"
    }


def test_llm_failure(monkeypatch):
    def fake_generate_answer(question, context):
        raise Exception("LLM unavailable")

    monkeypatch.setattr(
        "app.generate_answer",
        fake_generate_answer
    )

    response = client.post(
        "/ask",
        json={"question": "What is RAG?"}
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The language model service is currently unavailable."
    )


def test_source_metadata(monkeypatch):
    def fake_generate_answer(question, context):
        return "Docker runs applications in containers."

    monkeypatch.setattr(
        "app.generate_answer",
        fake_generate_answer
    )

    response = client.post(
        "/ask",
        json={"question": "What is Docker used for?"}
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["sources"]) >= 1
    assert data["sources"][0]["source"] == "knowledge.txt"
    assert "Docker" in data["sources"][0]["text"]


def test_sources_respect_threshold(monkeypatch):
    def fake_generate_answer(question, context):
        return "Docker answer."

    monkeypatch.setattr(
        "app.generate_answer",
        fake_generate_answer
    )

    response = client.post(
        "/ask",
        json={"question": "What is Docker used for?"}
    )

    data = response.json()

    for source in data["sources"]:
        assert source["score"] >= MIN_RETRIEVAL_SCORE