"""
JSON schema and Pydantic models for grounded generation responses.
"""
from __future__ import annotations
 
from typing import Literal
 
from pydantic import BaseModel, Field, field_validator
 
 
class Citation(BaseModel):
    chunk_id: str
    document_name: str
    section: str
    page: int
    excerpt: str
 
 
class GenerationResponse(BaseModel):
    answer_status: Literal["answered", "insufficient_context"]
    recommendation: str
    citations: list[Citation] = Field(default_factory=list)
    refusal_reason: str | None = None
    suggested_followups: list[str] = Field(default_factory=list)
 
    @field_validator("citations")
    @classmethod
    def citations_empty_on_refusal(cls, v: list[Citation], info) -> list[Citation]:
        status = info.data.get("answer_status")
        if status == "insufficient_context" and v:
            raise ValueError("citations must be empty when answer_status is insufficient_context")
        return v
 
    @field_validator("refusal_reason")
    @classmethod
    def refusal_reason_required_on_refusal(cls, v: str | None, info) -> str | None:
        status = info.data.get("answer_status")
        if status == "insufficient_context" and not v:
            raise ValueError("refusal_reason is required when answer_status is insufficient_context")
        if status == "answered" and v is not None:
            raise ValueError("refusal_reason must be null when answer_status is answered")
        return v
 
    @field_validator("suggested_followups")
    @classmethod
    def followups_empty_on_refusal(cls, v: list[str], info) -> list[str]:
        status = info.data.get("answer_status")
        if status == "insufficient_context" and v:
            raise ValueError("suggested_followups must be empty when answer_status is insufficient_context")
        return v[:3]
 
 
# OpenAI Structured Outputs schema (strict mode: all fields required, no extras).
RESPONSE_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "answer_status": {
            "type": "string",
            "enum": ["answered", "insufficient_context"],
        },
        "recommendation": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "document_name": {"type": "string"},
                    "section": {"type": "string"},
                    "page": {"type": "integer"},
                    "excerpt": {"type": "string"},
                },
                "required": ["chunk_id", "document_name", "section", "page", "excerpt"],
                "additionalProperties": False,
            },
        },
        "refusal_reason": {"type": ["string", "null"]},
        "suggested_followups": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 3,
        },
    },
    "required": ["answer_status", "recommendation", "citations", "refusal_reason", "suggested_followups"],
    "additionalProperties": False,
}
 
 
def validate_response(data: dict) -> GenerationResponse:
    """Validate a parsed model response; raises ValidationError on failure."""
    return GenerationResponse.model_validate(data)
 
 
def response_to_dict(response: GenerationResponse) -> dict:
    return response.model_dump()
