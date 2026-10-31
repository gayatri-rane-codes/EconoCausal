"""Unit tests for Phase 4 Day 1 FastAPI backend foundation."""
import unittest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.config import settings

class TestAPIFoundation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_endpoint_status_and_schema(self):
        """GET / should return 200 OK with correct service metadata."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], settings.PROJECT_NAME)
        self.assertEqual(data["version"], settings.VERSION)
        self.assertEqual(data["docs_url"], settings.DOCS_URL)
        self.assertEqual(data["health_url"], "/health")
        self.assertIn("Double Machine Learning", data["description"])

    def test_health_endpoint_status_and_schema(self):
        """GET /health should return 200 OK with healthy status and subsystem states."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["version"], settings.VERSION)
        self.assertIn("timestamp", data)
        self.assertIn("services", data)
        self.assertEqual(data["services"]["causal_engine"], "ready")
        self.assertEqual(data["services"]["optimization_service"], "ready")

    def test_openapi_json_available(self):
        """GET /openapi.json should return 200 OK with valid schema definitions."""
        response = self.client.get(settings.OPENAPI_URL)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["info"]["title"], settings.PROJECT_NAME)
        self.assertIn("/health", data["paths"])
        self.assertIn("/", data["paths"])

    def test_nonexistent_endpoint_returns_404(self):
        """GET /nonexistent should return 404 Not Found."""
        response = self.client.get("/nonexistent")
        self.assertEqual(response.status_code, 404)
        self.assertIn("detail", response.json())

if __name__ == "__main__":
    unittest.main()
