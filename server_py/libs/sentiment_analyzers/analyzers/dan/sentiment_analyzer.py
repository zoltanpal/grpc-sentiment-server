from libs.sentiment_analyzers.analyzers.base_analyzer import SentimentAnalyzerSingleton
from libs.sentiment_analyzers.models.sentiments import Sentiments


class DanishSentimentAnalyzer(SentimentAnalyzerSingleton):
    """
    A singleton class for analyzing Danish text sentiment using a pre-trained LLM.

    This class inherits from `SentimentAnalyzerSingleton` to ensure only one instance of the model
    is used for Danish sentiment analysis.
    """
    _instance = None  # Singleton instance

    def __new__(cls):
        """
        Creates and returns the singleton instance.
        The model is loaded only once when the first instance is created.

        Returns:
            DanishSentimentAnalyzer: The singleton instance of the sentiment analyzer.
        """
        if cls._instance is None:
            # If no instance exists, create one using the base class's __new__ method
            cls._instance = super().__new__(cls, "larskjeldgaard/senda")
        return cls._instance

    def analyze_text(self, text: str):
        """
        Analyzes the sentiment of a given text by calling the base class's `analyze` method
        to perform sentiment analysis, processes the results, and returns them in the `Sentiments` format.

        Args:
            text (str): The Danish text to analyze for sentiment.
        Returns:
            Sentiments: A `Sentiments` object containing the processed sentiment results.
        """
        sentiment_prediction = self.analyze(text)
        sentiment_results = {
            item["label"]: round(item["score"], 4) for item in sentiment_prediction[0]
        }

        # Rename the keys
        sentiment_results["negative"] = sentiment_results["negativ"]
        sentiment_results["positive"] = sentiment_results["positiv"]

        # Delete the unnecessary key-value pairs
        del sentiment_results["negativ"]
        del sentiment_results["positiv"]

        return Sentiments(**sentiment_results)