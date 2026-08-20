"""
Lightweight intent check so greetings and small talk skip guideline retrieval.

Medical / H. pylori questions always fall through to the RAG pipeline.
"""
from __future__ import annotations

import re
from datetime import datetime

Intent = str  # "greeting" | "casual" | "medical"

_MEDICAL_RE = re.compile(
    r"\b("
    r"pylori|helicobacter|h\.\s*pylori|"
    r"antibiotic|antibiotics|therapy|therapies|treatment|treatments|"
    r"regimen|bismuth|bqt|clarithromycin|amoxicillin|metronidazole|"
    r"tetracycline|rifabutin|levofloxacin|ppi|pcab|"
    r"eradication|infection|gastric|ulcer|dyspepsia|"
    r"dose|dosing|duration|guideline|diagnosis|testing|"
    r"penicillin|allergy|pregnant|pregnancy|salvage|naive|"
    r"probiotic|urea|breath"
    r")\b",
    re.IGNORECASE,
)

_GREETING_OPENERS = re.compile(
    r"^\s*("
    r"good\s+(morning|afternoon|evening|night)|"
    r"hello|hi+|hey+|howdy|greetings|yo"
    r")\b",
    re.IGNORECASE,
)

_HOW_ARE_YOU = re.compile(
    r"\bhow\s+(are|r)\s+(you|u)\b|\bhow'?s\s+it\s+going\b|\bwhat'?s\s+up\b",
    re.IGNORECASE,
)

_CASUAL_ONLY = re.compile(
    r"^\s*("
    r"thanks|thank\s+you|thx|ty|"
    r"ok|okay|k|"
    r"bye|goodbye|see\s+you|cya|"
    r"cool|great|nice|"
    r"yes|no|yep|nope"
    r")[\s!.?]*$",
    re.IGNORECASE,
)

_PUNCT = re.compile(r"[^\w\s.']+", re.UNICODE)


def classify_intent(text: str) -> Intent:
    """Return greeting, casual, or medical (RAG). Default is medical."""
    q = (text or "").strip()
    if not q:
        return "greeting"
    if _MEDICAL_RE.search(q):
        return "medical"

    compact = _PUNCT.sub(" ", q)
    compact = re.sub(r"\s+", " ", compact).strip()
    word_count = len(compact.split()) if compact else 0

    if _CASUAL_ONLY.match(compact):
        return "casual"

    greeting = bool(_GREETING_OPENERS.match(compact) or _HOW_ARE_YOU.search(compact))
    if greeting and word_count <= 12:
        return "greeting"

    if word_count <= 2 and _GREETING_OPENERS.match(compact):
        return "greeting"

    return "medical"


def _time_of_day_greeting(now: datetime | None = None) -> str:
    hour = (now or datetime.now()).hour
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    return "Good evening"


def first_conversation_greeting(now: datetime | None = None) -> str:
    return (
        f"{_time_of_day_greeting(now)}! 👋 I'm your H. pylori information assistant. "
        "How can I help you today?"
    )


def casual_reply(text: str, now: datetime | None = None) -> str:
    q = (text or "").strip()
    lower = q.lower()
    how_are = bool(_HOW_ARE_YOU.search(q))
    opener = (_GREETING_OPENERS.match(q) or _GREETING_OPENERS.match(_PUNCT.sub(" ", q)))

    if how_are and opener:
        return "Hi! I'm doing well, thanks. How can I help you?"
    if how_are:
        return "I'm doing well, thanks. How can I help you?"

    if re.match(r"^\s*good\s+morning\b", lower):
        return "Good morning! How can I help you today?"
    if re.match(r"^\s*good\s+evening\b", lower):
        return "Good evening! How can I help you today?"
    if re.match(r"^\s*good\s+afternoon\b", lower):
        return "Good afternoon! How can I help you today?"
    if re.match(r"^\s*hello\b", lower):
        return "Hello! What would you like to know about H. pylori?"
    if re.match(r"^\s*(hi+|hey+)\b", lower):
        return "Hi! How can I help you today?"

    if re.match(r"^\s*(thanks|thank you|thx|ty)\b", lower):
        return "You're welcome. Ask whenever you have a question about H. pylori."
    if re.match(r"^\s*(bye|goodbye|see you|cya)\b", lower):
        return "Goodbye! Take care."
    if re.match(r"^\s*(ok|okay|k|cool|great|nice|yes|yep)\b", lower):
        return "Sounds good. What would you like to know about H. pylori?"

    return f"{_time_of_day_greeting(now)}! How can I help you today?"


def chitchat_pipeline_result(query: str, intent: Intent) -> dict:
    """Structured API payload that skips retrieval and citations."""
    reply = casual_reply(query)
    return {
        "recommendation": reply,
        "excerpt": [],
        "evidence": [],
        "citation": [],
        "citations": [],
        "reranked_documents": [],
        "confidence": "high",
        "answer_status": intent,
        "refusal_reason": None,
        "suggested_followups": [],
        "_meta": {
            "llm_called": False,
            "skipped_retrieval": True,
            "intent": intent,
        },
    }
