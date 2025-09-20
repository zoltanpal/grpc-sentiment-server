from typing import List

from libs.sentiment_analyzers.analyzers.base_analyzer import SentimentAnalyzerSingleton
from libs.sentiment_analyzers.models.sentiments import Sentiments, LABEL_MAPPING_ROBERTA


class HungarianSentimentAnalyzer(SentimentAnalyzerSingleton):
    """
    A singleton class for analyzing Hungarian text sentiment using a pre-trained LLM.

    This class inherits from `SentimentAnalyzerSingleton` to ensure only one instance of the model
    is used for Hungarian sentiment analysis.
    """

    _instance = None  # Singleton instance

    def __new__(cls):
        """
        Creates and returns the singleton instance.
        The model is loaded only once when the first instance is created.

        Returns:
            HungarianSentimentAnalyzer: The singleton instance of the sentiment analyzer.
        """
        if cls._instance is None:
            # If no instance exists, create one using the base class's __new__ method
            cls._instance = super().__new__(
                cls, "NYTK/sentiment-hts5-xlm-roberta-hungarian"
            )
        return cls._instance

    def analyze_text(self, text: str) -> Sentiments:
        """
        Analyzes the sentiment of a given text by calling the base class's `analyze` method
        to perform sentiment analysis, processes the results, and returns them in the `Sentiments` format.

        Args:
            text (str): The Hungarian text to analyze for sentiment.
        Returns:
            Sentiments: A `Sentiments` object containing the processed sentiment results.
        """
        sentiment_prediction = self.analyze(text)
        sentiment_results = {
            LABEL_MAPPING_ROBERTA[item["label"]]: round(item["score"], 4)
            for item in sentiment_prediction[0]
        }
        return Sentiments(**sentiment_results)

    def analyze_batch(self, texts: List[str]) -> List[Sentiments]:
        predictions = self.pipeline(texts)
        return [
            Sentiments(
                **{
                    LABEL_MAPPING_ROBERTA[item["label"]]: round(item["score"], 4)
                    for item in prediction
                }
            )
            for prediction in predictions
        ]
