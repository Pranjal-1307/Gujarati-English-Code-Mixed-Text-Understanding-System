"""Optional multilingual sentence embedding generation."""
from __future__ import annotations

from typing import Sequence


class CodeMixedEmbedder:
    """Lazy-loading Sentence Transformers wrapper for code-mixed text."""
    def __init__(self, model_name: str = "sentence-transformers/LaBSE", device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Install sentence-transformers to generate embeddings") from exc
            kwargs = {"device": self.device} if self.device else {}
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    def encode(self, texts: str | Sequence[str], normalize_embeddings: bool = True):
        """Encode one sentence or a batch as dense NumPy vectors."""
        return self._load().encode(texts, convert_to_numpy=True, normalize_embeddings=normalize_embeddings, show_progress_bar=False)
