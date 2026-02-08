# libs/sentiment_analyzers/analyzers/dan/sentiment_analyzer.py

from __future__ import annotations

from typing import List

from libs.sentiment_analyzers.analyzers.base_analyzer import SentimentAnalyzerSingleton
from libs.sentiment_analyzers.models.sentiments import Sentiments


LABEL_MAPPING_DANISH = {
    # If the model outputs Danish labels:
    "positiv": "positive",
    "negativ": "negative",
    "neutral": "neutral",

    # If it outputs generic labels:
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",

    # If it outputs English labels:
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
}


class DanishSentimentAnalyzer(SentimentAnalyzerSingleton):
    # Let the base singleton handle instantiation/caching
    model_name: str = "larskjeldgaard/senda"
    truncation: bool = True
    top_k: int = 3  # or None / omit if your base supports it

    def analyze_text(self, text: str) -> Sentiments:
        preds = self.pipeline(text)[0]  # list[{"label":..., "score":...}, ...]
        return self._map_predictions_to_sentiments(preds, LABEL_MAPPING_DANISH)

    def analyze_batch(self, texts: List[str]) -> List[Sentiments]:
        preds_batch = self.pipeline(texts)
        return self._map_batch_predictions_to_sentiments(preds_batch, LABEL_MAPPING_DANISH)
