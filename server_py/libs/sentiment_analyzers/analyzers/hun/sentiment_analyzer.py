# libs/sentiment_analyzers/analyzers/hun/sentiment_analyzer.py

from typing import List

from libs.sentiment_analyzers.analyzers.base_analyzer import SentimentAnalyzerSingleton
from libs.sentiment_analyzers.models.sentiments import Sentiments

from libs.sentiment_analyzers.models.sentiments import LABEL_MAPPING_ROBERTA


class HungarianSentimentAnalyzer(SentimentAnalyzerSingleton):
    """
    Hungarian sentiment analyzer based on NYTK's RoBERTa model.
    Inherits singleton behavior from SentimentAnalyzerSingleton, so the
    HuggingFace pipeline is initialized once and reused.
    """

    # Model config (must be class attributes for base class to pick up)
    model_name: str = "NYTK/sentiment-hts5-xlm-roberta-hungarian"
    truncation: bool = True
    top_k: int = 3

    def analyze_text(self, text: str) -> Sentiments:
        """
        Analyze a single Hungarian text and return sentiment scores.
        """
        predictions = self.pipeline(text)[0]  # pipeline returns [[{label, score}, ...]]
        return Sentiments(
            **{
                LABEL_MAPPING_ROBERTA[item["label"]]: round(item["score"], 4)
                for item in predictions
            }
        )

    def analyze_batch(self, texts: List[str]) -> List[Sentiments]:
        """
        Analyze a batch of texts in one forward pass for efficiency.
        """
        predictions_batch = self.pipeline(texts)
        results: List[Sentiments] = []
        for predictions in predictions_batch:
            results.append(
                Sentiments(
                    **{
                        LABEL_MAPPING_ROBERTA[item["label"]]: round(item["score"], 4)
                        for item in predictions
                    }
                )
            )
        return results
