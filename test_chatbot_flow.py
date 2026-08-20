"""
End-to-end integration test for the chatbot pipeline flow:
Question -> Retrieval -> Reranking -> Recommendation + Excerpt + Citation -> Details
"""
import os
import sys
import unittest
from fastapi.testclient import TestClient

# Project-local HF cache setup
_HF_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache")
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", _HF_CACHE)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from app import app

class TestChatbotFlowEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)

    def test_flow_q1_first_line_treatment(self):
        """Test Question 1: First-line H. pylori treatment"""
        payload = {
            "query": "What is the preferred first-line treatment for treatment-naive patients when antibiotic susceptibility is unknown?",
            "top_k": 5,
            "pipeline": "rrf_rerank",
            "relevance_threshold": 0.35,
            "use_llm": True
        }
        res = self.client.post("/api/query", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        print("\n" + "="*70)
        print("TEST 1: First-line H. pylori Treatment")
        print("="*70)
        print("RECOMMENDATION:", data.get("recommendation"))
        print("EXCERPTS COUNT:", len(data.get("evidence", [])))
        print("CITATIONS:", [f"{c['document_id']}: {c['section']} (p.{c.get('page')})" for c in data.get("citations", [])])
        print("RERANKED DOCS:", len(data.get("reranked_documents", [])))
        if data.get("reranked_documents"):
            top_doc = data["reranked_documents"][0]
            print(f"TOP DOC #{top_doc['rank']} | Chunk: {top_doc['chunk_id']} | Relevance: {top_doc.get('score')} | Raw Logit: {top_doc.get('raw_score')}")

        # Assertions
        self.assertEqual(data["answer_status"], "answered")
        self.assertTrue(len(data["recommendation"]) > 20)
        self.assertTrue(len(data["evidence"]) >= 1)
        self.assertTrue(len(data["citations"]) >= 1)
        self.assertTrue(len(data["reranked_documents"]) >= 1)
        # Verify true reranker scores exist
        self.assertIsNotNone(data["reranked_documents"][0].get("score"))
        self.assertIsNotNone(data["reranked_documents"][0].get("raw_score"))

    def test_flow_q2_salvage_treatment(self):
        """Test Question 2: Salvage treatment after failure"""
        payload = {
            "query": "Which salvage regimens are recommended for treatment-experienced patients with persistent H. pylori infection?",
            "top_k": 5,
            "pipeline": "rrf_rerank",
            "relevance_threshold": 0.35,
            "use_llm": True
        }
        res = self.client.post("/api/query", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        print("\n" + "="*70)
        print("TEST 2: Salvage Treatment After Failure")
        print("="*70)
        print("RECOMMENDATION:", data.get("recommendation"))
        print("CITATIONS:", [c.get("chunk_id") for c in data.get("citations", [])])

        self.assertEqual(data["answer_status"], "answered")
        self.assertTrue(len(data["evidence"]) >= 1)
        self.assertTrue(len(data["citations"]) >= 1)

    def test_flow_q3_penicillin_allergy(self):
        """Test Question 3: Treatment for penicillin allergy"""
        payload = {
            "query": "How should H. pylori be managed in patients with a confirmed penicillin allergy?",
            "top_k": 5,
            "pipeline": "rrf_rerank",
            "relevance_threshold": 0.35,
            "use_llm": True
        }
        res = self.client.post("/api/query", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        print("\n" + "="*70)
        print("TEST 3: Treatment for Penicillin Allergy")
        print("="*70)
        print("RECOMMENDATION:", data.get("recommendation"))
        print("CITATIONS:", [f"{c['chunk_id']}: {c['section']}" for c in data.get("citations", [])])

        self.assertEqual(data["answer_status"], "answered")
        self.assertTrue(len(data["evidence"]) >= 1)
        self.assertTrue(len(data["citations"]) >= 1)

    def test_flow_q4_confirm_eradication(self):
        """Test Question 4: When to confirm eradication"""
        payload = {
            "query": "How long after completing therapy should post-treatment testing to confirm H. pylori eradication be performed?",
            "top_k": 5,
            "pipeline": "rrf_rerank",
            "relevance_threshold": 0.35,
            "use_llm": True
        }
        res = self.client.post("/api/query", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        print("\n" + "="*70)
        print("TEST 4: Confirming Eradication (Test of Cure)")
        print("="*70)
        print("RECOMMENDATION:", data.get("recommendation"))
        print("CITATIONS:", [f"{c['chunk_id']}: {c['section']}" for c in data.get("citations", [])])

        self.assertEqual(data["answer_status"], "answered")
        self.assertTrue(len(data["evidence"]) >= 1)
        self.assertTrue(len(data["citations"]) >= 1)

if __name__ == "__main__":
    unittest.main()
