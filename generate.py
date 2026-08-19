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
import time
import random
from difflib import SequenceMatcher
from typing import Any, Callable

import jsonschema
from openai import OpenAI, RateLimitError, APIConnectionError
from pydantic import ValidationError

DEFAULT_MAX_RETRIES = 2

RATE_LIMIT_MAX_RETRIES = 3


def _backoff_sleep(attempt: int) -> None:
    """Exponential backoff with jitter: ~2s, ~4s, ~8s..."""
    delay = (2 ** attempt) + random.uniform(0, 1)
    time.sleep(delay)

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


# Sentence-initial imperative clinical verbs ("Take 500mg...", "Start therapy...")
# and second-person directive phrasing ("you should...") — the model is meant
# to describe what the guideline states, never instruct the reader to act.
_DIRECTIVE_SENTENCE_START = re.compile(
    r"(?:^|[.!?]\s+)(Take|Start|Administer|Prescribe|Discontinue|Begin|Stop|Give|Use)\b",
    re.IGNORECASE,
)
_DIRECTIVE_PHRASES = re.compile(
    r"\byou should\b|\byou must\b|\byou need to\b|\bi recommend you\b",
    re.IGNORECASE,
)


def check_directive_language(text: str) -> list[str]:
    """
    Non-blocking tone guardrail. Flags phrasing that instructs the reader to
    act (prescriptive) rather than describing what the guideline says
    (descriptive). This is a transparency/telemetry layer, not a filter —
    it never rewrites or rejects the answer, since a heuristic false
    positive shouldn't silently break a correctly grounded response.
    Findings are surfaced in _meta.tone_warnings for inspection.
    """
    warnings: list[str] = []
    for match in _DIRECTIVE_SENTENCE_START.finditer(text):
        verb = match.group(1)
        warnings.append(f"Imperative sentence-start detected: \"{verb}...\"")
    for match in _DIRECTIVE_PHRASES.finditer(text):
        warnings.append(f"Directive phrase detected: \"{match.group(0)}\"")
    return warnings


def _excerpt_matches_chunk(excerpt: str, chunk_text: str, min_ratio: float) -> bool:
    excerpt_norm = _normalize_text(excerpt)
    chunk_norm = _normalize_text(chunk_text)
    if not excerpt_norm:
        return False
    if excerpt_norm in chunk_norm:
        return True
    if len(excerpt_norm) >= 20 and excerpt_norm[:20] in chunk_norm:
        return True

    # Below this point the excerpt is not a verbatim substring (the model
    # lightly reworded it) — measure how much of the EXCERPT is covered by
    # matching material in the chunk, rather than a plain SequenceMatcher
    # .ratio(), which divides by (len(excerpt) + len(chunk)) combined. That
    # symmetric ratio unfairly fails a short, legitimately-grounded excerpt
    # pulled from a much longer chunk — e.g. a 90-char excerpt against a
    # 600-char chunk scores ~0.10 even when every word of the excerpt
    # appears in the chunk, because the chunk's extra length dilutes the
    # ratio. What we actually want to know is "how much of the excerpt
    # itself is grounded," not "how similar are the two texts overall."
    matcher = SequenceMatcher(None, excerpt_norm, chunk_norm)
    total_matched = sum(block.size for block in matcher.get_matching_blocks())
    coverage = total_matched / len(excerpt_norm)
    return coverage >= min_ratio


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


def assemble_messages(
    query: str,
    chunks: list[dict],
    history: list[dict] | None = None,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": build_system_prompt(chunks)}]
    if history:
        for turn in history:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": query})
    return messages


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
            preview = cite.excerpt[:160] + ("..." if len(cite.excerpt) > 160 else "")
            warnings.append(
                f"Stripped citation {cite.chunk_id}: excerpt not grounded in chunk text "
                f"— model excerpt was: \"{preview}\""
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
    history: list[dict] | None = None,
) -> dict:
    api_client = client or OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("GENERATION_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    model_name = model or os.environ.get("GENERATION_MODEL", DEFAULT_MODEL)
    messages = assemble_messages(query, chunks, history=history)

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
        reasoning_effort="low",
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
    history: list[dict] | None = None,
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
    rate_limit_attempts = 0
 
    for attempt in range(max_retries + 1):
        try:
            metadata["llm_called"] = True
            parsed = caller(query, retrieved_chunks, client=client, model=model, history=history)
            validate_schema_raw(parsed)
            response = validate_response(parsed)
            response, warnings = verify_citations(
                response,
                retrieved_chunks,
                excerpt_min_ratio=excerpt_min_ratio,
            )
            metadata["citation_warnings"] = warnings
 
            if response.answer_status == "answered" and not response.citations:
                # Verification stripped every citation the model produced.
                # Give it another attempt (same budget as malformed-JSON
                # retries) before refusing outright — a single imperfect
                # excerpt shouldn't sink an otherwise well-grounded answer.
                if attempt < max_retries:
                    last_error = ValueError(
                        "All citations failed grounding verification; retrying."
                    )
                    continue
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
            if response.answer_status == "answered":
                metadata["tone_warnings"] = check_directive_language(response.recommendation)
            else:
                metadata["tone_warnings"] = []
            metadata["reasoning_effort"] = "low"
            metadata["model"] = model or os.environ.get("GENERATION_MODEL", DEFAULT_MODEL)
            result["_meta"] = metadata
            return result
 
        except RateLimitError as exc:
            # Groq free-tier TPM limit hit — back off and retry instead of
            # crashing. This is an infrastructure hiccup, not a grounding
            # failure, so it gets its own retry budget separate from
            # max_retries (which is for malformed-JSON / schema failures).
            last_error = exc
            rate_limit_attempts += 1
            metadata["rate_limited"] = True
            if rate_limit_attempts <= RATE_LIMIT_MAX_RETRIES:
                _backoff_sleep(rate_limit_attempts)
                continue
            break
 
        except APIConnectionError as exc:
            # Transient network issue — brief retry, no long backoff needed.
            last_error = exc
            rate_limit_attempts += 1
            if rate_limit_attempts <= RATE_LIMIT_MAX_RETRIES:
                time.sleep(1.5)
                continue
            break
 
        except (ValidationError, jsonschema.ValidationError, ValueError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
 
    # Every retry path exhausted — fail SAFE instead of raising, so the
    # FastAPI endpoint returns a normal refusal response (200) instead of
    # crashing with a 500 mid-demo.
    if isinstance(last_error, RateLimitError):
        reason = (
            "The system is temporarily busy (rate limit reached). "
            "Please wait a few seconds and try again."
        )
    elif isinstance(last_error, APIConnectionError):
        reason = "Could not reach the language model service. Please try again."
    else:
        reason = f"Generation failed after retries: {last_error}"
 
    result = build_refusal_response(reason)
    result["_meta"] = metadata
    return result
