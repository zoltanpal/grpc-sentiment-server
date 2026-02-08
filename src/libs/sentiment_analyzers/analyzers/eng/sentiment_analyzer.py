import threading

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from libs.sentiment_analyzers.models.sentiments import Sentiments


class EnglishSentimentAnalyzer:
    """
    A class for analyzing English text sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner).

    This class is responsible for performing sentiment analysis on English text using the VADER sentiment
    analysis tool. It ensures that the SentimentIntensityAnalyzer is initialized only once and can be used
    for multiple sentiment analyses.
    """
    _instance = None  # Singleton instance of the EnglishSentimentAnalyzer
    _lock = threading.Lock()  # Lock to ensure thread-safety when initializing the instance

    def __new__(cls):
        """
        Creates and returns the singleton instance of the EnglishSentimentAnalyzer.
        The initialization occurs only once, ensuring that the SentimentIntensityAnalyzer is reused.

        Returns:
            EnglishSentimentAnalyzer: The singleton instance of the English sentiment analyzer.
        """
        with cls._lock:
            if cls._instance is None:
                # If no instance exists, create one and initialize the SentimentIntensityAnalyzer
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        Initializes the SentimentIntensityAnalyzer.
        This method checks if the VADER lexicon is already downloaded. If not, it downloads the lexicon.
        """
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon")

        # Initialize the SentimentIntensityAnalyzer for sentiment analysis
        self.sid = SentimentIntensityAnalyzer()

    def analyze_text(self, text: str) -> Sentiments:
        """
        Analyzes the sentiment of a given English text using VADER.
        This method uses the SentimentIntensityAnalyzer to compute sentiment scores for the input text.

        Args:
            text (str): The English text to analyze for sentiment.
        Returns:
            Sentiments: A Sentiments object containing the polarity scores for negative, neutral,
                        positive, and compound sentiments.
        """
        if not text:
            raise ValueError("Missing text to analyze")  # Ensure that text is provided

        # Get sentiment polarity scores for the text
        results = self.sid.polarity_scores(text)

        # Return sentiment results as a Sentiments object
        return Sentiments(
            negative=results.get("neg", 0.0),
            neutral=results.get("neu", 0.0),
            positive=results.get("pos", 0.0),
            compound=results.get("compound", 0.0),
        )

    def analyze_batch(self, texts: list) -> list:
        """
        Analyzes the sentiment of multiple English texts using VADER.
        Processes all texts efficiently.

        Args:
            texts (list): List of English texts to analyze for sentiment.
        Returns:
            list: List of Sentiments objects with sentiment scores.
        """
        results = []
        for text in texts:
            if not text:
                results.append(Sentiments())
                continue
            
            # Get sentiment polarity scores for the text
            polarity = self.sid.polarity_scores(text)
            
            results.append(
                Sentiments(
                    negative=polarity.get("neg", 0.0),
                    neutral=polarity.get("neu", 0.0),
                    positive=polarity.get("pos", 0.0),
                    compound=polarity.get("compound", 0.0),
                )
            )
        return results
