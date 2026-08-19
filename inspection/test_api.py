"""
Tests for the FastAPI backend endpoints (/health, /api/query, /api/gold-questions).
"""
from __future__ import annotations

import unittest
from fastapi.testclient import TestClient

from app import app, get_base_and_chunks, get_reranker


class TestFastAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["index_loaded"])
        self.assertEqual(data["num_chunks"], 103)
        self.assertIn("pipeline", data)

    def test_gold_questions_endpoint(self):
        response = self.client.get("/api/gold-questions")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total"], 30)
        self.assertEqual(len(data["questions"]), 30)
        self.assertIn("expect_sections", data["questions"][0])

    def test_query_endpoint_first_line(self):
        payload = {
            "query": "What is the preferred first-line treatment for treatment-naive patients when antibiotic susceptibility is unknown?",
            "top_k": 5,
            "pipeline": "rrf_rerank",
            "relevance_threshold": 0.35,
            "use_llm": False,  # Test deterministic offline response
        }
        response = self.client.post("/api/query", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify structured shape
        self.assertIn("recommendation", data)
        self.assertIn("evidence", data)
        self.assertIn("citations", data)
        self.assertIn("reranked_documents", data)
        self.assertIn("confidence", data)
        self.assertIn("answer_status", data)
        self.assertIn("latency_ms", data)
        self.assertEqual(data["pipeline_used"], "rrf_rerank")

        # Verify reranked documents structure & actual reranker scores
        reranked = data["reranked_documents"]
        self.assertEqual(len(reranked), 5)
        top_doc = reranked[0]
        self.assertEqual(top_doc["rank"], 1)
        self.assertIn("chunk_id", top_doc)
        self.assertIn("document", top_doc)
        self.assertIn("section", top_doc)
        self.assertIn("excerpt", top_doc)
        self.assertIsNotNone(top_doc["score"])
        self.assertGreaterEqual(top_doc["score"], 0.0)
        self.assertLessEqual(top_doc["score"], 1.0)
        self.assertIsNotNone(top_doc["raw_score"])

    def test_query_endpoint_refusal_on_empty(self):
        response = self.client.post("/api/query", json={"query": "   "})
        self.assertIn(response.status_code, [400, 422])


if __name__ == "__main__":
    unittest.main()
