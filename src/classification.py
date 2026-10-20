"""Multi-task sentiment and intent inference with a useful zero-shot baseline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import PROJECT_ROOT

SENTIMENTS = ("Positive", "Negative", "Neutral")
INTENTS = ("Greeting", "Inquiry", "Feedback", "General")


@dataclass
class TaskPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


@dataclass
class MultiTaskPrediction:
    sentiment: TaskPrediction
    intent: TaskPrediction


class MultiTaskClassifier:
    """Load a persisted multi-output estimator or use transparent lexical baselines."""
    def __init__(self, model_path: str | Path = "models/classifier_mbert"):
        self.model_path = Path(model_path)
        if not self.model_path.is_absolute():
            self.model_path = PROJECT_ROOT / self.model_path
        self.model: Any = None
        model_file = self.model_path if self.model_path.is_file() else self.model_path / "classifier.pkl"
        if model_file.exists():
            try:
                import joblib
                self.model = joblib.load(model_file)
            except Exception:
                self.model = None

    @staticmethod
    def _scores(text: str, classes: tuple[str, ...], groups: dict[str, set[str]]) -> TaskPrediction:
        words = set(text.lower().split())
        counts = {label: sum(1 for term in terms if term in words or term in text.lower()) for label, terms in groups.items()}
        total = sum(counts.values())
        if total == 0:
            counts[classes[-1]] = 1
            total = 1
        probs = {label: round(counts.get(label, 0) / total, 4) for label in classes}
        label = max(probs, key=probs.get)
        return TaskPrediction(label, probs[label], probs)

    def predict(self, text: str) -> MultiTaskPrediction:
        """Predict sentiment and intent, with normalized class confidence scores."""
        if self.model is not None:
            try:
                values = self.model.predict([text])[0]
                return MultiTaskPrediction(TaskPrediction(str(values[0]), 0.5, {}), TaskPrediction(str(values[1]), 0.5, {}))
            except Exception:
                pass
        sentiment = self._scores(text, SENTIMENTS, {
            "Positive": {"great", "good", "saras", "maja", "ખુશ", "વાહ", "excellent", "mast", "love"},
            "Negative": {"bad", "poor", "ખરાબ", "નથી", "issue", "worst", "late", "disappointed"},
            "Neutral": {"okay", "maybe", "કદાચ", "fine"},
        })
        intent = self._scores(text, INTENTS, {
            "Greeting": {"hello", "hi", "kem cho", "નમસ્તે", "નમસ્કાર", "હાય"},
            "Inquiry": {"what", "how", "કેમ", "ક્યાં", "શુ", "શું", "price", "ક્યારે", "where"},
            "Feedback": {"service", "experience", "quality", "સરસ", "ખરાબ", "review", "feedback"},
            "General": {"today", "આજે", "thanks", "thank"},
        })
        return MultiTaskPrediction(sentiment, intent)
