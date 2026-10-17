from fastapi.testclient import TestClient

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

    assert len(data["sources"]) == 3

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