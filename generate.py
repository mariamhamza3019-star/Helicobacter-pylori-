"""
Grounded answer generation for the H. pylori clinical RAG pipeline.

Accepts any list of retrieved chunks matching the retrieval schema (with optional
``score`` for relevance gating). Calls openai/gpt-oss-120b with structured JSON
output, validates schema, and verifies citations against retrieved chunks.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import random
from difflib import SequenceMatcher
from typing import Any, Callable

import jsonschema
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError
from pydantic import ValidationError

from env_loader import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

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
DEFAULT_RAW_SCORE_THRESHOLD = float(os.environ.get("RAW_SCORE_THRESHOLD", "-2.0"))
DEFAULT_RELEVANCE_THRESHOLD = DEFAULT_RAW_SCORE_THRESHOLD
DEFAULT_NORMALIZED_FALLBACK_THRESHOLD = 0.35
DEFAULT_EXCERPT_MIN_RATIO = 0.72
DEFAULT_MAX_RETRIES = 2

REFUSAL_LOW_RELEVANCE = (
    "No retrieved chunks met the minimum relevance threshold for this query."
)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


# --- Tone Guard -----------------------------------------------------------
_DIRECTIVE_SENTENCE_START = re.compile(
    r"(?:^|[.!?]\s+)(Take|Start|Administer|Prescribe|Discontinue|Begin|Stop|Give|Use)\b",
    re.IGNORECASE,
)
_DIRECTIVE_PHRASES = re.compile(
    r"\byou should\b|\byou must\b|\byou need to\b|\bi recommend you\b",
    re.IGNORECASE,
)
_NON_CAUTIOUS_PHRASES = re.compile(
    r"\bguaranteed\b|\bdefinitely\b|\bcertainly\b|\bwithout(?: any)? exception\b"
    r"|\balways works\b|\bnever fails\b",
    re.IGNORECASE,
)


def check_tone_guard(text: str) -> list[str]:
    """
    Non-blocking Tone Guard. Flags phrasing that is either (a) prescriptive
    or (b) overconfident/absolute. Surfaced in _meta.tone_warnings.
    """
    warnings: list[str] = []
    for match in _DIRECTIVE_SENTENCE_START.finditer(text):
        verb = match.group(1)
        warnings.append(f"Imperative sentence-start detected: \"{verb}...\"")
    for match in _DIRECTIVE_PHRASES.finditer(text):
        warnings.append(f"Directive phrase detected: \"{match.group(0)}\"")
    for match in _NON_CAUTIOUS_PHRASES.finditer(text):
        warnings.append(f"Overconfident/non-cautious phrasing detected: \"{match.group(0)}\"")
    return warnings


# Backward-compatible alias for the old name.
check_directive_language = check_tone_guard


# --- Medical Safety Gate ---------------------------------------------------
_UNSUPPORTED_DIAGNOSIS = re.compile(
    r"\byou (?:have|are (?:infected with|suffering from|positive for))\b"
    r"[^.!?]{0,60}?\b(h\.?\s*pylori|helicobacter|peptic ulcer|gastric cancer|infection)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_TREATMENT_OUTCOME = re.compile(
    r"\bwill (?:cure|eradicate)\b|\bguarantee(?:d|s)? to (?:cure|eradicate|work)\b"
    r"|\b100% (?:cure|success|effective)\b|\bcompletely (?:cures|eradicates)\b",
    re.IGNORECASE,
)


def check_medical_safety_claims(text: str) -> list[str]:
    """
    Non-blocking Medical Safety Gate. Flags unsupported clinical CONTENT.
    Surfaced in _meta.safety_warnings.
    """
    warnings: list[str] = []
    for match in _UNSUPPORTED_DIAGNOSIS.finditer(text):
        warnings.append(f"Unsupported diagnostic claim about the reader detected: \"{match.group(0)}\"")
    for match in _UNSUPPORTED_TREATMENT_OUTCOME.finditer(text):
        warnings.append(f"Unsupported treatment-outcome guarantee detected: \"{match.group(0)}\"")
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

    matcher = SequenceMatcher(None, excerpt_norm, chunk_norm)
    total_matched = sum(block.size for block in matcher.get_matching_blocks())
    coverage = total_matched / len(excerpt_norm)
    return coverage >= min_ratio


def _chunk_lookup(chunks: list[dict]) -> dict[str, dict]:
    return {c["chunk_id"]: c for c in chunks}


def _top_relevance_score(chunks: list[dict]) -> float | None:
    scores = [c["score"] for c in chunks if c.get("score") is not None]
    return max(scores) if scores else None


def _top_raw_score(chunks: list[dict]) -> float | None:
    scores = [c["raw_score"] for c in chunks if c.get("raw_score") is not None]
    return max(scores) if scores else None


def should_refuse_low_relevance(
    chunks: list[dict],
    threshold: float = DEFAULT_RAW_SCORE_THRESHOLD,
) -> bool:
    if not chunks:
        return True
    top_raw = _top_raw_score(chunks)
    if top_raw is not None:
        return top_raw < threshold
    top_norm = _top_relevance_score(chunks)
    if top_norm is None:
        return False
    return top_norm < DEFAULT_NORMALIZED_FALLBACK_THRESHOLD


def gate_metadata(chunks: list[dict], threshold: float) -> dict[str, Any]:
    top = _top_raw_score(chunks)
    if top is None:
        top = _top_relevance_score(chunks)
    return {
        "evidence_retrieved": bool(chunks),
        "relevance_passed": bool(chunks) and top is not None and top >= threshold,
        "top_raw_score": top,
    }


def low_relevance_reason(chunks: list[dict], threshold: float) -> str:
    if not chunks:
        return "No chunks were retrieved for this query."
    top = _top_relevance_score(chunks)
    if top is None:
        return REFUSAL_LOW_RELEVANCE
    return (
        f"{REFUSAL_LOW_RELEVANCE} "
        f"(top score {top:.3f} < threshold {threshold:.3f})."
    )


def build_refusal_response(reason: str) -> dict:
    response = GenerationResponse(
        answer_status="insufficient_context",
        recommendation=(
            "I couldn't find sufficiently relevant evidence in the H. pylori "
            "guidelines to answer this reliably."
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
        timeout=45.0,
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
    allow_fallback: bool = True,
    fallback_model_fn: Callable[..., dict] | None = None,
) -> dict:
    """Generate a grounded answer from retrieved chunks."""
    metadata: dict[str, Any] = {"citation_warnings": [], "llm_called": False}
    metadata.update(gate_metadata(retrieved_chunks, relevance_threshold))

    if should_refuse_low_relevance(retrieved_chunks, relevance_threshold):
        reason = low_relevance_reason(retrieved_chunks, relevance_threshold)
        result = build_refusal_response(reason)
        # Evaluate guards directly on refusal text to prevent misleading UI outputs
        metadata["tone_warnings"] = check_tone_guard(result["recommendation"])
        metadata["safety_warnings"] = check_medical_safety_claims(result["recommendation"])
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
                if attempt < max_retries:
                    last_error = ValueError(
                        "All citations failed grounding verification; retrying."
                    )
                    continue
                metadata["citations_unusable"] = True
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
            
            # Evaluate tone and safety guards on the recommendation text regardless of answer_status
            metadata["tone_warnings"] = check_tone_guard(response.recommendation)
            metadata["safety_warnings"] = check_medical_safety_claims(response.recommendation)
            metadata["reasoning_effort"] = "low"
            metadata["model"] = model or os.environ.get("GENERATION_MODEL", DEFAULT_MODEL)
            result["_meta"] = metadata

            if metadata.get("citations_unusable"):
                fallback = _maybe_gemini_fallback(
                    query,
                    retrieved_chunks,
                    excerpt_min_ratio=excerpt_min_ratio,
                    max_retries=max_retries,
                    history=history,
                    allow_fallback=allow_fallback,
                    call_model_fn=call_model_fn,
                    fallback_model_fn=fallback_model_fn,
                )
                if fallback is not None:
                    return fallback
            return result

        except RateLimitError as exc:
            last_error = exc
            rate_limit_attempts += 1
            metadata["rate_limited"] = True
            if rate_limit_attempts <= RATE_LIMIT_MAX_RETRIES:
                _backoff_sleep(rate_limit_attempts)
                continue
            break

        except APIConnectionError as exc:
            last_error = exc
            rate_limit_attempts += 1
            if rate_limit_attempts <= RATE_LIMIT_MAX_RETRIES:
                time.sleep(1.5)
                continue
            break

        except (APITimeoutError, TimeoutError) as exc:
            last_error = exc
            logger.warning("Primary LLM timed out (%s)", type(exc).__name__)
            rate_limit_attempts += 1
            if rate_limit_attempts <= RATE_LIMIT_MAX_RETRIES:
                time.sleep(1.5)
                continue
            break

        except (ValidationError, jsonschema.ValidationError, ValueError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break

        except Exception as exc:
            last_error = exc
            logger.warning("Primary LLM failed (%s)", type(exc).__name__)
            break

    metadata["generation_failed"] = True
    metadata["primary_error"] = type(last_error).__name__ if last_error else "unknown"
    fallback = _maybe_gemini_fallback(
        query,
        retrieved_chunks,
        excerpt_min_ratio=excerpt_min_ratio,
        max_retries=max_retries,
        history=history,
        allow_fallback=allow_fallback,
        call_model_fn=call_model_fn,
        fallback_model_fn=fallback_model_fn,
    )
    if fallback is not None:
        return fallback

    from gemini_fallback import gemini_configured

    if allow_fallback and fallback_model_fn is not None:
        return both_llms_failed_response()
    if allow_fallback and call_model_fn is None and gemini_configured():
        return both_llms_failed_response()

    result = build_refusal_response(_user_safe_generation_error(last_error))
    metadata["tone_warnings"] = check_tone_guard(result["recommendation"])
    metadata["safety_warnings"] = check_medical_safety_claims(result["recommendation"])
    result["_meta"] = metadata
    return result


def _user_safe_generation_error(exc: Exception | None) -> str:
    if isinstance(exc, RateLimitError):
        return (
            "The system is temporarily busy (rate limit reached). "
            "Please wait a few seconds and try again."
        )
    if isinstance(exc, (APIConnectionError, APITimeoutError, TimeoutError, ConnectionError)):
        return "Could not reach the language model service. Please try again."
    return "The language model could not produce a grounded answer from the retrieved evidence."


def _is_grounded_success(result: dict) -> bool:
    return result.get("answer_status") == "answered" and bool(result.get("citations"))


def _is_evidence_refusal(result: dict) -> bool:
    if result.get("answer_status") != "insufficient_context":
        return False
    meta = result.get("_meta") or {}
    if meta.get("generation_failed"):
        return False
    return True


def _maybe_gemini_fallback(
    query: str,
    retrieved_chunks: list[dict],
    *,
    excerpt_min_ratio: float,
    max_retries: int,
    history: list[dict] | None,
    allow_fallback: bool,
    call_model_fn: Callable[..., dict] | None,
    fallback_model_fn: Callable[..., dict] | None,
) -> dict | None:
    if not allow_fallback:
        return None
    if call_model_fn is not None and fallback_model_fn is None:
        return None
    return _try_gemini_fallback(
        query,
        retrieved_chunks,
        excerpt_min_ratio=excerpt_min_ratio,
        max_retries=max_retries,
        history=history,
        fallback_model_fn=fallback_model_fn,
    )


def _try_gemini_fallback(
    query: str,
    retrieved_chunks: list[dict],
    *,
    excerpt_min_ratio: float,
    max_retries: int,
    history: list[dict] | None,
    fallback_model_fn: Callable[..., dict] | None,
) -> dict | None:
    caller = fallback_model_fn
    if caller is None:
        from gemini_fallback import call_gemini_model, gemini_configured

        if not gemini_configured():
            return None
        caller = call_gemini_model

    logger.warning("Primary LLM unusable; trying Gemini fallback")
    try:
        result = generate_answer(
            query,
            retrieved_chunks,
            excerpt_min_ratio=excerpt_min_ratio,
            max_retries=max_retries,
            call_model_fn=caller,
            history=history,
            allow_fallback=False,
        )
    except Exception as exc:
        logger.warning("Gemini fallback failed (%s)", type(exc).__name__)
        return None

    meta = result.setdefault("_meta", {})
    meta["fallback"] = "gemini"
    meta["model"] = os.environ.get("GEMINI_MODEL") or meta.get("model") or "gemini-3.5-flash-lite"
    if _is_grounded_success(result) or _is_evidence_refusal(result):
        meta["generation_failed"] = False
        return result
    logger.warning("Gemini fallback returned an unusable response")
    return None


def both_llms_failed_response() -> dict:
    result = build_refusal_response(
        "I couldn't generate an answer right now. Please try again in a moment."
    )
    result["_meta"] = {
        "citation_warnings": [],
        "llm_called": True,
        "generation_failed": True,
        "fallback": "gemini",
        "both_llms_failed": True,
        "tone_warnings": check_tone_guard(result["recommendation"]),
        "safety_warnings": check_medical_safety_claims(result["recommendation"]),
    }
    return result