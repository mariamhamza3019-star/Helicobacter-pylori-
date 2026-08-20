"""
Gemini Flash-Lite fallback generation. Uses the same RAG context and JSON schema
as the primary LLM. API key is read from GEMINI_API_KEY (never hardcoded).
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx

from generate import _parse_model_content, assemble_messages
from schema import RESPONSE_JSON_SCHEMA

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_URL_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
GEMINI_TIMEOUT_S = 45.0


def gemini_configured() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


def _gemini_model_name() -> str:
    return os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


def _messages_to_gemini(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        text = msg.get("content") or ""
        if role == "system":
            system_parts.append(text)
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": text}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": text}]})
    if not contents or contents[0]["role"] != "user":
        contents.insert(0, {"role": "user", "parts": [{"text": "Answer using only the retrieved context."}]})
    return "\n\n".join(system_parts), contents


_TYPE_MAP = {
    "object": "OBJECT",
    "array": "ARRAY",
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "null": "NULL",
}


def _gemini_schema(schema: Any) -> Any:
    """Convert JSON Schema to Gemini responseSchema (OpenAPI / protobuf types)."""
    if isinstance(schema, list):
        return [_gemini_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    out: dict[str, Any] = {}
    nullable = False
    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if t != "null"]
        nullable = "null" in raw_type
        raw_type = non_null[0] if non_null else "string"

    for key, value in schema.items():
        if key in ("additionalProperties", "maxItems"):
            continue
        if key == "type":
            continue
        out[key] = _gemini_schema(value)

    if raw_type is not None:
        mapped = _TYPE_MAP.get(str(raw_type).lower())
        out["type"] = mapped or str(raw_type).upper()
    if nullable:
        out["nullable"] = True
    return out


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def call_gemini_model(
    query: str,
    chunks: list[dict],
    *,
    client: Any = None,
    model: str | None = None,
    history: list[dict] | None = None,
) -> dict:
    """
    Call Gemini with retrieved chunks only. Signature matches ``call_model``.
    ``client`` is ignored (OpenAI client is not used here).
    """
    del client  # primary-LLM client is not applicable
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    model_name = model or _gemini_model_name()
    messages = assemble_messages(query, chunks, history=history)
    system_text, contents = _messages_to_gemini(messages)
    url = GEMINI_URL_TMPL.format(model=model_name)
    payload = {
        "system_instruction": {"parts": [{"text": system_text}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "responseSchema": _gemini_schema(RESPONSE_JSON_SCHEMA),
        },
    }

    try:
        response = httpx.post(
            url,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=GEMINI_TIMEOUT_S,
        )
    except httpx.TimeoutException as exc:
        logger.warning("Gemini fallback timed out (%s)", type(exc).__name__)
        raise TimeoutError("Gemini request timed out") from exc
    except httpx.HTTPError as exc:
        logger.warning("Gemini fallback connection error (%s)", type(exc).__name__)
        raise ConnectionError("Could not reach Gemini") from exc

    if response.status_code >= 400:
        try:
            err = response.json().get("error", {})
            message = err.get("message") or response.text[:300]
        except Exception:
            message = response.text[:300]
        logger.warning("Gemini fallback HTTP %s: %s", response.status_code, message)
        raise RuntimeError(f"Gemini request failed with status {response.status_code}")

    try:
        body = response.json()
    except json.JSONDecodeError as exc:
        raise ValueError("Gemini returned non-JSON envelope") from exc

    try:
        text = body["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Gemini returned an empty or unusable payload") from exc

    if not text or not str(text).strip():
        raise ValueError("Gemini returned empty content")

    return _parse_model_content(_strip_fences(str(text)))
