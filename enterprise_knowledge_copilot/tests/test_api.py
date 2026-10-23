from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_documents_list_shape():
    response = client.get("/documents")
    assert response.status_code == 200
    assert "documents" in response.json()
