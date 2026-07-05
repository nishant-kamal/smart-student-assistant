from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_qa_endpoint():
    r = client.post(
        "/qa",
        json={"context": "The sky is blue.", "question": "What color is the sky?"},
    )
    assert r.status_code == 200
    assert "answer" in r.json()


def test_summarize_endpoint():
    r = client.post("/summarize", json={"text": "Some long text to summarize."})
    assert r.status_code == 200
    assert "summary" in r.json()
