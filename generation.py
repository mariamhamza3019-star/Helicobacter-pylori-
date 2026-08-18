"""
Grounded generation layer - Qwen 3.6-27B via Groq API.
Single function to generate answers with citations, grounded in retrieved evidence.
"""

import json
import logging
import os
from typing import Optional

from groq import Groq

logger = logging.getLogger(__name__)

GROQ_MODEL = "qwen/qwen3.6-27b"
TEMPERATURE = 0.2
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a medical AI for H. pylori. Output ONLY valid JSON, nothing else.

CRITICAL: Your response must be ONLY a JSON object. No markdown, no explanation, no code blocks.

Schema:
{
  "recommendation": "string or empty",
  "evidence_excerpt": "string or empty", 
  "citations": [{"chunk_id": "string", "section": "string", "page": number}],
  "refused": boolean,
  "refusal_reason": "string or null"
}

Rules:
1. Answer ONLY from provided evidence - do NOT use external knowledge
2. If you cannot answer from evidence, set refused=true and explain why
3. Citations MUST reference only chunk_ids in the provided evidence
4. Every field must be present
5. Output starts with { and ends with } - no other text before or after
"""


def generate_answer(question: str, retrieved_chunks: list[dict]) -> dict:
    """
    Generate grounded answer with citations.
    
    Args:
        question: User question
        retrieved_chunks: List of chunks with chunk_id, section, page, text
    
    Returns:
        {
            "recommendation": str or "",
            "evidence_excerpt": str or "",
            "citations": list or [],
            "refused": bool,
            "refusal_reason": str or None,
            "validated": bool,
            "validation_errors": list
        }
    """
    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "recommendation": "",
            "evidence_excerpt": "",
            "citations": [],
            "refused": True,
            "refusal_reason": "GROQ_API_KEY not set",
            "validated": False,
            "validation_errors": ["Missing API key"],
        }

    # Check chunks
    if not retrieved_chunks:
        return {
            "recommendation": "",
            "evidence_excerpt": "",
            "citations": [],
            "refused": True,
            "refusal_reason": "No evidence retrieved",
            "validated": False,
            "validation_errors": ["No chunks provided"],
        }

    # Format evidence
    evidence_lines = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        chunk_id = chunk.get("chunk_id", "?")
        section = chunk.get("section", "?")
        subsection = chunk.get("subsection", "")
        page = chunk.get("page_start") or chunk.get("page", "?")
        text = chunk.get("text", "")
        
        header = f"[{i}] {section}"
        if subsection:
            header += f" / {subsection}"
        header += f" (id={chunk_id}, p.{page})"
        
        evidence_lines.append(header)
        evidence_lines.append(text)
        evidence_lines.append("")
    
    evidence_text = "\n".join(evidence_lines)

    # Call Groq
    user_message = f"""Question: {question}

Evidence:
{evidence_text}

Respond with ONLY JSON. No markdown, no explanation."""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            reasoning_effort="none",
        )
        response_text = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return {
            "recommendation": "",
            "evidence_excerpt": "",
            "citations": [],
            "refused": True,
            "refusal_reason": f"API error: {str(e)}",
            "validated": False,
            "validation_errors": [str(e)],
        }

    # Parse JSON
    result = _parse_json(response_text)
    if result is None:
        return {
            "recommendation": "",
            "evidence_excerpt": "",
            "citations": [],
            "refused": True,
            "refusal_reason": "Invalid JSON response",
            "validated": False,
            "validation_errors": ["Failed to parse JSON"],
        }

    # Validate citations
    validation_errors = _validate_citations(result, retrieved_chunks)
    
    if validation_errors:
        result["recommendation"] = ""
        result["evidence_excerpt"] = ""
        result["citations"] = []
        result["refused"] = True
        result["refusal_reason"] = "Hallucinated citations detected"
    
    result["validated"] = len(validation_errors) == 0
    result["validation_errors"] = validation_errors
    
    return result


def _parse_json(text: str) -> Optional[dict]:
    """Try to parse JSON, handling markdown wrappers and extra text."""
    text = text.strip()
    original_text = text  # Keep for debugging
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Remove markdown markers
    if text.startswith("```json"):
        text = text[7:].lstrip()
    elif text.startswith("```python"):
        text = text[9:].lstrip()
    elif text.startswith("```"):
        text = text[3:].lstrip()
    
    if text.endswith("```"):
        text = text[:-3].rstrip()
    
    # Try again after stripping markdown
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Extract JSON object (find first { and last })
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Log what we got for debugging
    logger.error(f"Could not parse JSON. Model returned:\n{original_text[:300]}")
    return None


def _validate_citations(result: dict, chunks: list[dict]) -> list:
    """Check all citations reference chunks actually provided."""
    valid_ids = {chunk.get("chunk_id") for chunk in chunks}
    errors = []
    
    for citation in result.get("citations", []):
        chunk_id = citation.get("chunk_id")
        if chunk_id not in valid_ids:
            errors.append(f"Citation '{chunk_id}' not in retrieved chunks")
    
    return errors
