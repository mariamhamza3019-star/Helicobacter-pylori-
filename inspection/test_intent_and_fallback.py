"""
Offline tests for greeting routing and Gemini fallback (no live API calls).
"""
from __future__ import annotations

import unittest

from generate import generate_answer, DEFAULT_RELEVANCE_THRESHOLD
from intent_router import classify_intent, casual_reply, chitchat_pipeline_result

ACG_CHUNK_1 = {
    "chunk_id": "ACG_0001",
    "document_id": "ACG_2024",
    "text": (
        "For treatment-naive patients with H. pylori infection, bismuth quadruple therapy "
        "(BQT) for 14 days is the preferred regimen when antibiotic susceptibility is unknown. "
        "Rifabutin triple therapy or potassium-competitive acid blocker dual therapy for "
        "14 days is a suitable empiric alternative in patients without penicillin allergy."
    ),
    "page": 1,
    "section": "ABSTRACT",
    "source": "ACG Clinical Guideline 2024: Treatment of Helicobacter pylori Infection",
    "topic": "Helicobacter pylori",
    "score": 0.82,
}

LOW_RELEVANCE_CHUNK = {
    "chunk_id": "ACG_0099",
    "document_id": "ACG_2024",
    "text": "SUPPLEMENTARY MATERIAL accompanies this paper at http://links.lww.com/AJG/D362.",
    "page": 1,
    "section": "FRONT MATTER",
    "source": "ACG Clinical Guideline 2024: Treatment of Helicobacter pylori Infection",
    "topic": "Helicobacter pylori",
    "score": 0.12,
}


def _mock_answered_bqt(query: str, chunks: list[dict], **kwargs) -> dict:
    return {
        "answer_status": "answered",
        "recommendation": (
            "For treatment-naive patients when susceptibility is unknown, bismuth quadruple "
            "therapy (BQT) for 14 days is the preferred regimen."
        ),
        "citations": [
            {
                "chunk_id": "ACG_0001",
                "document_name": ACG_CHUNK_1["source"],
                "section": "ABSTRACT",
                "page": 1,
                "excerpt": (
                    "bismuth quadruple therapy (BQT) for 14 days is the preferred regimen when "
                    "antibiotic susceptibility is unknown"
                ),
            }
        ],
        "refusal_reason": None,
        "suggested_followups": [],
    }


def _fail_llm(*_args, **_kwargs):
    raise ConnectionError("primary unavailable")


def _fail_gemini(*_args, **_kwargs):
    raise TimeoutError("gemini timed out")


class TestIntentRouter(unittest.TestCase):
    def test_good_morning(self):
        self.assertEqual(classify_intent("Good morning"), "greeting")
        self.assertIn("Good morning", casual_reply("Good morning"))
        self.assertIn("How can I help you today?", casual_reply("Good morning"))

    def test_hi_how_are_you(self):
        self.assertEqual(classify_intent("Hi, how are you?"), "greeting")
        self.assertIn("I'm doing well", casual_reply("Hi, how are you?"))

    def test_hello(self):
        self.assertEqual(classify_intent("Hello"), "greeting")
        self.assertIn("H. pylori", casual_reply("Hello"))

    def test_medical_what_is_h_pylori(self):
        self.assertEqual(classify_intent("What is H. pylori?"), "medical")

    def test_medical_recommended_treatment(self):
        self.assertEqual(classify_intent("What is the recommended treatment?"), "medical")

    def test_medical_antibiotics(self):
        self.assertEqual(classify_intent("What antibiotics are recommended?"), "medical")

    def test_greeting_plus_medical_goes_to_rag(self):
        self.assertEqual(
            classify_intent("Hi, what is the recommended treatment for H. pylori?"),
            "medical",
        )

    def test_chitchat_payload_skips_retrieval(self):
        result = chitchat_pipeline_result("Good morning", "greeting")
        self.assertEqual(result["answer_status"], "greeting")
        self.assertEqual(result["citations"], [])
        self.assertTrue(result["_meta"]["skipped_retrieval"])


class TestGeminiFallback(unittest.TestCase):
    def test_primary_success_does_not_call_gemini(self):
        called = {"n": 0}

        def gemini_fn(*args, **kwargs):
            called["n"] += 1
            return _mock_answered_bqt(*args, **kwargs)

        result = generate_answer(
            "What is first-line therapy?",
            [ACG_CHUNK_1],
            call_model_fn=_mock_answered_bqt,
            fallback_model_fn=gemini_fn,
        )
        self.assertEqual(result["answer_status"], "answered")
        self.assertEqual(called["n"], 0)
        self.assertNotEqual(result["_meta"].get("fallback"), "gemini")

    def test_primary_failure_uses_gemini(self):
        result = generate_answer(
            "What is first-line therapy?",
            [ACG_CHUNK_1],
            call_model_fn=_fail_llm,
            fallback_model_fn=_mock_answered_bqt,
            max_retries=0,
        )
        self.assertEqual(result["answer_status"], "answered")
        self.assertEqual(result["_meta"].get("fallback"), "gemini")
        self.assertGreaterEqual(len(result["citations"]), 1)

    def test_both_llms_fail(self):
        result = generate_answer(
            "What is first-line therapy?",
            [ACG_CHUNK_1],
            call_model_fn=_fail_llm,
            fallback_model_fn=_fail_gemini,
            max_retries=0,
        )
        self.assertEqual(result["answer_status"], "insufficient_context")
        self.assertTrue(result["_meta"].get("both_llms_failed"))
        self.assertIn("try again", result["refusal_reason"].lower())
        self.assertNotIn("GEMINI_API_KEY", result["recommendation"])
        self.assertNotIn("GEMINI_API_KEY", result["refusal_reason"] or "")

    def test_low_relevance_does_not_call_gemini(self):
        called = {"n": 0}

        def gemini_fn(*args, **kwargs):
            called["n"] += 1
            return _mock_answered_bqt(*args, **kwargs)

        result = generate_answer(
            "What is the recommended H. pylori regimen in pregnancy?",
            [LOW_RELEVANCE_CHUNK],
            relevance_threshold=DEFAULT_RELEVANCE_THRESHOLD,
            call_model_fn=_fail_llm,
            fallback_model_fn=gemini_fn,
        )
        self.assertEqual(result["answer_status"], "insufficient_context")
        self.assertFalse(result["_meta"]["llm_called"])
        self.assertEqual(called["n"], 0)
        self.assertEqual(result["citations"], [])


if __name__ == "__main__":
    unittest.main()
