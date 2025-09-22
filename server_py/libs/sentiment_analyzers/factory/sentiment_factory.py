from __future__ import annotations

import threading
from typing import Dict

from libs.sentiment_analyzers.analyzers.dan.sentiment_analyzer import DanishSentimentAnalyzer
from libs.sentiment_analyzers.analyzers.eng.sentiment_analyzer import EnglishSentimentAnalyzer
from libs.sentiment_analyzers.analyzers.hun.sentiment_analyzer import HungarianSentimentAnalyzer


class SentimentAnalyzerFactory:
    """
    Lazy, cached access to language-specific sentiment analyzers.

    Usage:
        analyzer = SentimentAnalyzerFactory.get_analyzer("hun")
        result = analyzer.analyze_text("...")
    """

    _lock = threading.Lock()
    _analyzers: Dict[str, object] = {}

    _constructors = {
        "hun": HungarianSentimentAnalyzer,
        "dan": DanishSentimentAnalyzer,
        "eng": EnglishSentimentAnalyzer,
    }

    @staticmethod
    def get_analyzer(language: str):
        """
        Retrieve (and lazily create) the analyzer for a language.
        Supported: 'hun', 'dan', 'eng'.
        """
        if language not in SentimentAnalyzerFactory._constructors:
            raise ValueError(f"Unsupported language: {language}")

        # Fast path: already cached
        analyzer = SentimentAnalyzerFactory._analyzers.get(language)
        if analyzer is not None:
            return analyzer

        # Slow path: create once
        with SentimentAnalyzerFactory._lock:
            analyzer = SentimentAnalyzerFactory._analyzers.get(language)
            if analyzer is None:
                ctor = SentimentAnalyzerFactory._constructors[language]
                analyzer = ctor()  # __init__/__new__ should load pipeline once
                SentimentAnalyzerFactory._analyzers[language] = analyzer
        return analyzer
