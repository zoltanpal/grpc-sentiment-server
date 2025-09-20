from libs.sentiment_analyzers.analyzers.dan.sentiment_analyzer import \
    DanishSentimentAnalyzer
from libs.sentiment_analyzers.analyzers.eng.sentiment_analyzer import \
    EnglishSentimentAnalyzer
from libs.sentiment_analyzers.analyzers.hun.sentiment_analyzer import \
    HungarianSentimentAnalyzer


class SentimentAnalyzerFactory:
    """
    A factory class that provides access to language-specific sentiment analyzers.

    This class is used to get the appropriate sentiment analyzer instance based on the specified language.
    It supports English (eng), Danish (dan), and Hungarian (hun) analyzers.
    """
    _analyzers = {
        "hun": HungarianSentimentAnalyzer(),
        "dan": DanishSentimentAnalyzer(),
        "eng": EnglishSentimentAnalyzer(),
    }

    @staticmethod
    def get_analyzer(language: str):
        """
        Retrieves the sentiment analyzer for the specified language.

        Args:
            language (str): The language code for which the sentiment analyzer is requested.
                            Supported values are 'hun' (Hungarian), 'dan' (Danish), and 'eng' (English).
        Returns:
            SentimentAnalyzer: The language-specific sentiment analyzer instance.
        """
        if language not in SentimentAnalyzerFactory._analyzers:
            raise ValueError(f"Unsupported language: {language}")
        return SentimentAnalyzerFactory._analyzers[language]