"""Text cleanup and informal spelling normalization."""
from __future__ import annotations

import re
from pathlib import Path

from .utils import load_json

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_RE = re.compile(r"<[^>]+>")
MENTION_RE = re.compile(r"(?<!\w)@[\w.]+")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
EMOJI_RE = re.compile("[\\U0001F1E0-\\U0001FAFF\\U00002600-\\U000027BF\\uFE0F\\u200D]")


def clean_text(text: str) -> str:
    """Remove URLs, HTML, and mentions while normalizing whitespace."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"\s+([,!?;:.])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def normalize_informal_words(text: str, dict_path: str | Path = "models/normalization_rules.json") -> str:
    """Replace known informal tokens without changing their surrounding punctuation."""
    rules = load_json(dict_path)
    if not isinstance(rules, dict):
        raise ValueError("Normalization rules must be a JSON object")
    pattern = re.compile(r"\b[\w']+\b", re.UNICODE)
    return pattern.sub(lambda match: str(rules.get(match.group(0).lower(), match.group(0))), text)


def preprocess_pipeline(text: str, dict_path: str | Path = "models/normalization_rules.json") -> str:
    """Clean and normalize a code-mixed message."""
    return normalize_informal_words(clean_text(text), dict_path)
