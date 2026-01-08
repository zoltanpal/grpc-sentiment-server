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

        scores = {}
        for item in preds:
            raw_label = item["label"]
            label = LABEL_MAPPING_DANISH.get(raw_label, raw_label)
            scores[label] = round(float(item["score"]), 4)

        # Ensure Sentiments always has expected keys (optional, depends on your Sentiments model)
        scores.setdefault("negative", 0.0)
        scores.setdefault("neutral", 0.0)
        scores.setdefault("positive", 0.0)

        return Sentiments(**scores)

    def analyze_batch(self, texts: List[str]) -> List[Sentiments]:
        preds_batch = self.pipeline(texts)

        out: List[Sentiments] = []
        for preds in preds_batch:
            scores = {}
            for item in preds:
                raw_label = item["label"]
                label = LABEL_MAPPING_DANISH.get(raw_label, raw_label)
                scores[label] = round(float(item["score"]), 4)

            scores.setdefault("negative", 0.0)
            scores.setdefault("neutral", 0.0)
            scores.setdefault("positive", 0.0)

            out.append(Sentiments(**scores))
        return out
