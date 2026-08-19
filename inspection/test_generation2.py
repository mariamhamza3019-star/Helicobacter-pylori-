"""
Tests for the grounded generation layer (offline — no live API calls).
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from generate import (
    DEFAULT_RELEVANCE_THRESHOLD,
    build_refusal_response,
    generate_answer,
    should_refuse_low_relevance,
    verify_citations,
)
from schema import Citation, GenerationResponse, validate_response

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

ACG_CHUNK_3 = {
    "chunk_id": "ACG_0003",
    "document_id": "ACG_2024",
    "text": (
        "H. pylori remains one of the most common chronic bacterial infections of humans "
        "worldwide. It is the leading cause of infection-associated cancer globally and is "
        "categorized by the World Health Organization International Agency for Research on "
        "Cancer as a group I (definite) carcinogen because of its causal association with "
        "gastric cancer."
    ),
    "page": 2,
    "section": "INTRODUCTION",
    "source": "ACG Clinical Guideline 2024: Treatment of Helicobacter pylori Infection",
    "topic": "Helicobacter pylori",
    "score": 0.71,
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

IRRELEVANT_TOPIC_CHUNK = {
    "chunk_id": "ACG_0040",
    "document_id": "ACG_2024",
    "text": "The guideline panel members were selected based on their clinical expertise.",
    "page": 2,
    "section": "METHODS",
    "source": "ACG Clinical Guideline 2024: Treatment of Helicobacter pylori Infection",
    "topic": "Helicobacter pylori",
    "score": 0.55,
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
        "suggested_followups": [
            "What are the alternative first-line regimens if a patient has a penicillin allergy?",
        ],
    }


def _mock_answered_carcinogen(query: str, chunks: list[dict], **kwargs) -> dict:
    return {
        "answer_status": "answered",
        "recommendation": (
            "H. pylori is classified as a group I (definite) carcinogen due to its causal "
            "association with gastric cancer."
        ),
        "citations": [
            {
                "chunk_id": "ACG_0003",
                "document_name": ACG_CHUNK_3["source"],
                "section": "INTRODUCTION",
                "page": 2,
                "excerpt": (
                    "categorized by the World Health Organization International Agency for "
                    "Research on Cancer as a group I (definite) carcinogen"
                ),
            }
        ],
        "refusal_reason": None,
        "suggested_followups": [
            "What surveillance is recommended after H. pylori eradication in high-risk patients?",
        ],
    }


def _mock_refusal_pregnancy(query: str, chunks: list[dict], **kwargs) -> dict:
    return {
        "answer_status": "insufficient_context",
        "recommendation": (
            "The retrieved guideline excerpts do not provide enough relevant information "
            "to answer this question safely."
        ),
        "citations": [],
        "refusal_reason": "No retrieved chunks address dosing for pregnant patients.",
        "suggested_followups": [],
    }


def _mock_hallucinated_citation(query: str, chunks: list[dict], **kwargs) -> dict:
    return {
        "answer_status": "answered",
        "recommendation": "Some fabricated recommendation.",
        "citations": [
            {
                "chunk_id": "ACG_FAKE_9999",
                "document_name": "ACG Clinical Guideline 2024",
                "section": "ABSTRACT",
                "page": 1,
                "excerpt": "This chunk does not exist in the retrieved set.",
            },
            {
                "chunk_id": "ACG_0001",
                "document_name": ACG_CHUNK_1["source"],
                "section": "ABSTRACT",
                "page": 1,
                "excerpt": "bismuth quadruple therapy (BQT) for 14 days is the preferred regimen",
            },
        ],
        "refusal_reason": None,
        "suggested_followups": [],
    }


class TestRelevanceGuard(unittest.TestCase):
    def test_refuses_when_top_score_below_threshold(self):
        self.assertTrue(should_refuse_low_relevance([LOW_RELEVANCE_CHUNK]))
        self.assertTrue(
            should_refuse_low_relevance([LOW_RELEVANCE_CHUNK], threshold=0.35)
        )

    def test_passes_when_top_score_above_threshold(self):
        self.assertFalse(should_refuse_low_relevance([ACG_CHUNK_1, LOW_RELEVANCE_CHUNK]))


class TestSchemaValidation(unittest.TestCase):
    def test_valid_answered_response(self):
        payload = _mock_answered_bqt("", [])
        response = validate_response(payload)
        self.assertEqual(response.answer_status, "answered")
        self.assertEqual(len(response.citations), 1)

    def test_refusal_requires_reason(self):
        with self.assertRaises(Exception):
            validate_response(
                {
                    "answer_status": "insufficient_context",
                    "recommendation": "Cannot answer.",
                    "citations": [],
                    "refusal_reason": None,
                }
            )


class TestCitationVerification(unittest.TestCase):
    def test_strips_unknown_chunk_id(self):
        response = GenerationResponse(
            answer_status="answered",
            recommendation="Test",
            citations=[
                Citation(
                    chunk_id="ACG_FAKE_9999",
                    document_name="ACG",
                    section="ABSTRACT",
                    page=1,
                    excerpt="fabricated",
                ),
                Citation(
                    chunk_id="ACG_0001",
                    document_name=ACG_CHUNK_1["source"],
                    section="ABSTRACT",
                    page=1,
                    excerpt="bismuth quadruple therapy (BQT) for 14 days",
                ),
            ],
            refusal_reason=None,
        )
        cleaned, warnings = verify_citations(response, [ACG_CHUNK_1])
        self.assertEqual(len(cleaned.citations), 1)
        self.assertEqual(cleaned.citations[0].chunk_id, "ACG_0001")
        self.assertTrue(any("unknown chunk_id" in w for w in warnings))


class TestGenerateAnswer(unittest.TestCase):
    def test_query1_first_line_treatment_answered_with_citations(self):
        result = generate_answer(
            "What is the preferred first-line treatment for treatment-naive H. pylori "
            "when susceptibility is unknown?",
            [ACG_CHUNK_1],
            call_model_fn=_mock_answered_bqt,
        )
        self.assertEqual(result["answer_status"], "answered")
        self.assertGreaterEqual(len(result["citations"]), 1)
        self.assertIsNone(result["refusal_reason"])
        self.assertTrue(result["_meta"]["llm_called"])

    def test_query2_carcinogen_answered_with_citations(self):
        result = generate_answer(
            "How is H. pylori classified regarding gastric cancer risk?",
            [ACG_CHUNK_3],
            call_model_fn=_mock_answered_carcinogen,
        )
        self.assertEqual(result["answer_status"], "answered")
        self.assertEqual(result["citations"][0]["chunk_id"], "ACG_0003")

    def test_query3_low_relevance_short_circuits_without_llm(self):
        result = generate_answer(
            "What is the recommended H. pylori regimen in pregnancy?",
            [LOW_RELEVANCE_CHUNK],
            relevance_threshold=DEFAULT_RELEVANCE_THRESHOLD,
            call_model_fn=_mock_answered_bqt,
        )
        self.assertEqual(result["answer_status"], "insufficient_context")
        self.assertEqual(result["citations"], [])
        self.assertIsNotNone(result["refusal_reason"])
        self.assertFalse(result["_meta"]["llm_called"])

    def test_query4_out_of_scope_model_refusal(self):
        result = generate_answer(
            "What is the recommended H. pylori antibiotic dosing for pregnant patients?",
            [IRRELEVANT_TOPIC_CHUNK],
            call_model_fn=_mock_refusal_pregnancy,
        )
        self.assertEqual(result["answer_status"], "insufficient_context")
        self.assertEqual(result["citations"], [])
        self.assertIn("pregnant", result["refusal_reason"].lower())

    def test_query5_hallucinated_chunk_id_stripped(self):
        result = generate_answer(
            "What is the preferred first-line treatment for treatment-naive H. pylori?",
            [ACG_CHUNK_1],
            call_model_fn=_mock_hallucinated_citation,
        )
        self.assertEqual(result["answer_status"], "answered")
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["citations"][0]["chunk_id"], "ACG_0001")
        self.assertTrue(
            any("unknown chunk_id" in w for w in result["_meta"]["citation_warnings"])
        )


class TestRefusalHelper(unittest.TestCase):
    def test_build_refusal_response_shape(self):
        result = build_refusal_response("Test reason.")
        self.assertEqual(result["answer_status"], "insufficient_context")
        self.assertEqual(result["refusal_reason"], "Test reason.")


if __name__ == "__main__":
    unittest.main()
