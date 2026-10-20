"""Dictionary-assisted Romanized Gujarati transliteration."""
from __future__ import annotations

import re
from difflib import get_close_matches
from pathlib import Path

from .utils import load_json


class GujaratiTransliterator:
    """Map Romanized Gujarati words to Gujarati script using exact/fuzzy lookup."""
    def __init__(self, dictionary_path: str | Path = "data/dictionary/guj_eng_transliteration.json", cutoff: float = 0.82):
        self.mapping: dict[str, str] = {str(k).lower(): str(v) for k, v in load_json(dictionary_path).items()}
        self.cutoff = cutoff

    def transliterate_token(self, token: str) -> str:
        """Transliterate a token, retaining unknown words and punctuation."""
        key = token.lower()
        if key in self.mapping:
            result = self.mapping[key]
        elif re.fullmatch(r"[A-Za-z]+", token):
            match = get_close_matches(key, self.mapping.keys(), n=1, cutoff=self.cutoff)
            result = self.mapping[match[0]] if match else token
        else:
            result = token
        if token[:1].isupper() and result != token:
            return result
        return result

    def transliterate(self, text: str) -> str:
        """Transliterate each word while preserving whitespace and punctuation."""
        return re.sub(r"[A-Za-z]+", lambda match: self.transliterate_token(match.group(0)), text)


def transliterate_text(text: str, dictionary_path: str | Path = "data/dictionary/guj_eng_transliteration.json") -> str:
    """Convenience wrapper for the default transliteration dictionary."""
    return GujaratiTransliterator(dictionary_path).transliterate(text)
