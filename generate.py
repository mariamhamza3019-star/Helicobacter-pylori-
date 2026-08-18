"""
Grounded answer generation for the H. pylori clinical RAG pipeline.

Accepts any list of retrieved chunks matching the retrieval schema (with optional
``score`` for relevance gating). Calls openai/gpt-oss-120b with structured JSON
output, validates schema, and verifies citations against retrieved chunks.
"""
from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from typing import Any, Callable

import jsonschema
from openai import OpenAI
from pydantic import ValidationError

from grounding_system_prompt import build_system_prompt
from schema import (
    RESPONSE_JSON_SCHEMA,
    GenerationResponse,
    response_to_dict,
    validate_response,
)

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_RELEVANCE_THRESHOLD = 0.35
DEFAULT_EXCERPT_MIN_RATIO = 0.72
DEFAULT_MAX_RETRIES = 2

REFUSAL_LOW_RELEVANCE = (
    "No retrieved chunks met the minimum relevance threshold for this query."
)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _excerpt_matches_chunk(excerpt: str, chunk_text: str, min_ratio: float) -> bool:
    excerpt_norm = _normalize_text(excerpt)
    chunk_norm = _normalize_text(chunk_text)
    if not excerpt_norm:
        return False
    if excerpt_norm in chunk_norm:
        return True
    if len(excerpt_norm) >= 20 and excerpt_norm[:20] in chunk_norm:
        return True
    return SequenceMatcher(None, excerpt_norm, chunk_norm).ratio() >= min_ratio


def _chunk_lookup(chunks: list[dict]) -> dict[str, dict]:
    return {c["chunk_id"]: c for c in chunks}


def _top_relevance_score(chunks: list[dict]) -> float | None:
    scores = [c["score"] for c in chunks if c.get("score") is not None]
    return max(scores) if scores else None


def should_refuse_low_relevance(
    chunks: list[dict],
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> bool:
    """Pre-generation guard: refuse when top chunk score is below threshold."""
    if not chunks:
        return True
    top_score = _top_relevance_score(chunks)
    if top_score is None:
        return False
    return top_score < threshold


def build_refusal_response(reason: str) -> dict:
    response = GenerationResponse(
        answer_status="insufficient_context",
        recommendation=(
            "The retrieved guideline excerpts do not provide enough relevant "
            "information to answer this question safely."
        ),
        citations=[],
        refusal_reason=reason,
    )
    return response_to_dict(response)


def assemble_messages(query: str, chunks: list[dict]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt(chunks)},
        {"role": "user", "content": query},
    ]


def validate_schema_raw(data: dict) -> None:
    jsonschema.validate(instance=data, schema=RESPONSE_JSON_SCHEMA)


def verify_citations(
    response: GenerationResponse,
    chunks: list[dict],
    *,
    excerpt_min_ratio: float = DEFAULT_EXCERPT_MIN_RATIO,
) -> tuple[GenerationResponse, list[str]]:
    """
    Drop citations whose chunk_id is unknown or whose excerpt does not match
    the chunk text. Returns the cleaned response and a list of warning messages.
    """
    if response.answer_status != "answered":
        return response, []

    lookup = _chunk_lookup(chunks)
    kept: list = []
    warnings: list[str] = []

    for cite in response.citations:
        chunk = lookup.get(cite.chunk_id)
        if chunk is None:
            warnings.append(f"Stripped citation with unknown chunk_id: {cite.chunk_id}")
            continue
        if not _excerpt_matches_chunk(cite.excerpt, chunk.get("text", ""), excerpt_min_ratio):
            warnings.append(
                f"Stripped citation {cite.chunk_id}: excerpt not grounded in chunk text"
            )
            continue
        kept.append(cite)

    cleaned = response.model_copy(update={"citations": kept})
    return cleaned, warnings


def _parse_model_content(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned non-JSON content: {exc}") from exc


def call_model(
    query: str,
    chunks: list[dict],
    *,
    client: OpenAI | None = None,
    model: str | None = None,
) -> dict:
    api_client = client or OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("GENERATION_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    model_name = model or os.environ.get("GENERATION_MODEL", DEFAULT_MODEL)
    messages = assemble_messages(query, chunks)

    completion = api_client.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "grounded_clinical_answer",
                "strict": True,
                "schema": RESPONSE_JSON_SCHEMA,
            },
        },
        temperature=0.0,
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError("Model returned empty content")
    return _parse_model_content(content)


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    *,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    excerpt_min_ratio: float = DEFAULT_EXCERPT_MIN_RATIO,
    max_retries: int = DEFAULT_MAX_RETRIES,
    client: OpenAI | None = None,
    model: str | None = None,
    call_model_fn: Callable[..., dict] | None = None,
) -> dict:
    """
    Generate a grounded answer from retrieved chunks.

    Parameters
    ----------
    query : str
        Clinician question.
    retrieved_chunks : list[dict]
        Ranked chunks with keys: chunk_id, document_id, text, page, section,
        source, topic. Optional ``score`` enables the pre-generation relevance guard.
    """
    metadata: dict[str, Any] = {"citation_warnings": [], "llm_called": False}

    if should_refuse_low_relevance(retrieved_chunks, relevance_threshold):
        top = _top_relevance_score(retrieved_chunks)
        if not retrieved_chunks:
            reason = "No chunks were retrieved for this query."
        elif top is None:
            reason = REFUSAL_LOW_RELEVANCE
        else:
            reason = (
                f"{REFUSAL_LOW_RELEVANCE} "
                f"(top score {top:.3f} < threshold {relevance_threshold:.3f})."
            )
        result = build_refusal_response(reason)
        result["_meta"] = metadata
        return result

    caller = call_model_fn or call_model
    last_error: Exception | None = None
    parsed: dict | None = None

    for attempt in range(max_retries + 1):
        try:
            metadata["llm_called"] = True
            parsed = caller(query, retrieved_chunks, client=client, model=model)
            validate_schema_raw(parsed)
            response = validate_response(parsed)
            response, warnings = verify_citations(
                response,
                retrieved_chunks,
                excerpt_min_ratio=excerpt_min_ratio,
            )
            metadata["citation_warnings"] = warnings

            if response.answer_status == "answered" and not response.citations:
                if warnings:
                    response = GenerationResponse(
                        answer_status="insufficient_context",
                        recommendation=(
                            "Citation verification removed all supporting excerpts; "
                            "cannot provide a grounded answer."
                        ),
                        citations=[],
                        refusal_reason=(
                            "Model citations failed grounding checks against retrieved chunks."
                        ),
                    )
                else:
                    response = GenerationResponse(
                        answer_status="insufficient_context",
                        recommendation=(
                            "The retrieved context does not support a fully cited answer."
                        ),
                        citations=[],
                        refusal_reason="No valid citations were produced for the answer.",
                    )

            result = response_to_dict(response)
            result["_meta"] = metadata
            return result
        except (ValidationError, jsonschema.ValidationError, ValueError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break

    raise RuntimeError(
        f"Generation failed after {max_retries + 1} attempt(s): {last_error}"
    ) from last_error
