"""Word-level language identification for Gujarati-English text."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .utils import PROJECT_ROOT

GUJARATI_RE = re.compile(r"[\u0A80-\u0AFF]")
TOKEN_RE = re.compile(r"[\u0A80-\u0AFF]+|[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\w\s]", re.UNICODE)
GUJARATI_WORDS = {"kem", "cho", "chho", "che", "chhe", "majama", "maja", "saras", "bhai", "ben", "shu", "su", "nathi", "hu", "tame", "ame", "mara", "tamara", "pan", "have", "aaje", "kal", "ghar", "javu", "avjo", "aavjo", "mast", "saru", "karjo", "karu", "thai", "thay", "chhe"}


def is_gujarati_script(token: str) -> bool:
    """Return whether a token contains Gujarati Unicode characters."""
    return bool(GUJARATI_RE.search(token))


def _load_model(model_path: str | Path | None):
    if model_path is None:
        return None
    path = Path(model_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return None
    try:
        import joblib
        return joblib.load(path)
    except (ImportError, OSError, ValueError):
        return None


def identify_word(token: str, model=None) -> str:
    """Assign a language tag using script and lexicon heuristics, then model fallback."""
    if is_gujarati_script(token):
        return "GUJ_SCRIPT"
    if token.isdigit() or not any(ch.isalpha() for ch in token):
        return "SYMBOL_NUMBER"
    lower = token.lower()
    if lower in GUJARATI_WORDS:
        return "TRANSLITERATED_GUJ"
    if model is not None:
        try:
            return str(model.predict([lower])[0])
        except Exception:
            pass
    if re.fullmatch(r"[A-Za-z]+(?:['’][A-Za-z]+)?", token):
        return "ENG"
    return "SYMBOL_NUMBER"


def identify_languages(text: str, model_path: str | Path | None = "models/lid_model.pkl") -> list[tuple[str, str]]:
    """Return (token, tag) pairs for every word, number, and punctuation token."""
    model = _load_model(model_path)
    return [(token, identify_word(token, model)) for token in TOKEN_RE.findall(text)]


def train_lid_model(words: Iterable[str], labels: Iterable[str], output_path: str | Path = "models/lid_model.pkl") -> Path:
    """Train a character n-gram TF-IDF/RandomForest word tagger and persist it."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib

    model = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 5)), RandomForestClassifier(n_estimators=120, class_weight="balanced", random_state=42))
    model.fit(list(words), list(labels))
    path = Path(output_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path
