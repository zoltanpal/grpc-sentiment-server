# server_py/server.py
"""
gRPC Sentiment Service (Python)
==============================

This service exposes two unary RPCs:

- Analyze(AnalyzeRequest) -> AnalyzeResponse
- BatchAnalyze(BatchAnalyzeRequest) -> BatchAnalyzeResponse

It wraps your installed analyzer factory:
    libs.sentiment_analyzers.factory.sentiment_factory.SentimentAnalyzerFactory

Expected analyzer output (per text):
    {
      "negative": 0.7516,
      "very_negative": 0.1092,
      "neutral": 0.0977,
      "positive": 0.0357,
      "very_positive": 0.0059,
      "compound": -0.6144
    }

Environment variables:
- GRPC_HOST     (default: "127.0.0.1")
- GRPC_PORT     (default: "50051")
- MAX_WORKERS   (default: "4")

Run:
    python server_py/server.py
"""

from __future__ import annotations

import os
import time
import logging
from concurrent import futures
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

import grpc

import pb.sentiment_pb2 as pb
import pb.sentiment_pb2_grpc as pb_grpc

# Sentiment Analyzer Factory
from libs.sentiment_analyzers.factory.sentiment_factory import SentimentAnalyzerFactory  # type: ignore

# ---------------------------------------------------------------------------

SERVER_VERSION = "poc-0.3.0"

# Basic logging setup (adjust level/format as you wish)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("sentiment-grpc-server")


def _to_dict(obj: Any) -> dict:
    """
    Normalize various Python objects into a dictionary.

    Conversion order:
      - Mapping -> dict(...)
      - dataclass -> asdict(...)
      - pydantic v2 -> .model_dump()
      - pydantic v1 -> .dict()
      - generic objects -> vars(obj)
      - None/unknown -> {}
    """
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return obj.model_dump()  # pydantic v2
        except Exception:  # pragma: no cover - best-effort
            pass
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return obj.dict()  # pydantic v1
        except Exception:  # pragma: no cover - best-effort
            pass
    if hasattr(obj, "__dict__"):
        try:
            return vars(obj)
        except Exception:  # pragma: no cover - best-effort
            pass
    return {}


def map_result_to_response(text: str, result_obj: Any) -> pb.AnalyzeResponse:
    """
    Map analyzer output (dict like {"negative": 0.75, "positive": 0.03, "compound": -0.61, ...})
    to the gRPC AnalyzeResponse format:
      - title: original text
      - sentiment_key: dominant label (ignores 'compound')
      - sentiment_value: dominant score
      - sentiments: all scores as a map
    """
    result = _to_dict(result_obj)

    if not result:
        return pb.AnalyzeResponse(
            title=text,
            sentiment_key="unknown",
            sentiment_value=0.0,
            sentiments={}
        )

    # Determine dominant label (ignore compound)
    numeric_labels = {k: v for k, v in result.items() if k != "compound"}
    if numeric_labels:
        sentiment_key, sentiment_value = max(numeric_labels.items(), key=lambda kv: kv[1])
    else:
        sentiment_key, sentiment_value = "compound", float(result.get("compound", 0.0))

    # Build response
    return pb.AnalyzeResponse(
        title=text,
        sentiment_key=sentiment_key,
        sentiment_value=float(sentiment_value),
        sentiments={str(k): float(v) for k, v in result.items() if isinstance(v, (int, float))}
    )


# ---------------------------------------------------------------------------
# gRPC Service
# ---------------------------------------------------------------------------
class SentimentService(pb_grpc.SentimentServiceServicer):
    """Implements SentimentService RPCs using SentimentAnalyzerFactory."""

    def Analyze(self, request: pb.AnalyzeRequest, context: grpc.ServicerContext) -> pb.AnalyzeResponse:
        """
        Analyze a single text.

        - language: analyzer key, e.g., "hun", "dan", "eng".
        - On error, returns dominant_label="error" and sets gRPC status to INTERNAL.
        """
        try:
            lang = request.language or "hun"
            analyzer = SentimentAnalyzerFactory.get_analyzer(lang)
            raw = analyzer.analyze_text(request.text)
            return map_result_to_response(request.text, raw)
        except Exception as e:
            log.exception("Analyze failed (language=%r)", getattr(request, "language", None))
            context.set_details(f"Analyze error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb.AnalyzeResponse(text=request.text, dominant_label="error")


    def BatchAnalyze(self, request: pb.BatchAnalyzeRequest, context) -> pb.BatchAnalyzeResponse:
        """
        Analyze multiple texts, grouped by language. Uses analyzer.analyze_batch if available,
        falling back to per-text Analyze otherwise.
        """
        results = []

        # Group inputs by language
        by_lang: dict[str, list[str]] = {}
        for item in request.items:
            lang = item.language or "hun"
            by_lang.setdefault(lang, []).append(item.text)

        for lang, texts in by_lang.items():
            analyzer = SentimentAnalyzerFactory.get_analyzer(lang)
            try:
                if hasattr(analyzer, "analyze_batch") and callable(analyzer.analyze_batch):
                    # One forward pass for all texts
                    batch_out = analyzer.analyze_batch(texts)
                    for text, sentiments in zip(texts, batch_out):
                        results.append(map_result_to_response(text, sentiments))
                else:
                    # Fallback: process each text separately
                    for text in texts:
                        raw = analyzer.analyze_text(text)
                        results.append(map_result_to_response(text, raw))
            except Exception as e:
                log.exception("BatchAnalyze failed (language=%r)", lang)
                for text in texts:
                    results.append(
                        pb.AnalyzeResponse(
                            title=text,
                            sentiment_key="error",
                            sentiment_value=0.0,
                            sentiments={}
                        )
                    )

        return pb.BatchAnalyzeResponse(results=results)

# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------
def serve(host: str | None = None, port: int | None = None) -> None:
    """
    Start the gRPC server.

    Args:
        host: Host/IP to bind (default from $GRPC_HOST or "127.0.0.1").
        port: Port to bind (default from $GRPC_PORT or 50051).
    """
    host = host or os.getenv("GRPC_HOST", "127.0.0.1")
    port = port or int(os.getenv("GRPC_PORT", "50051"))
    max_workers = int(os.getenv("MAX_WORKERS", "4"))

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb_grpc.add_SentimentServiceServicer_to_server(SentimentService(), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()

    log.info(
        "🚀 Sentiment gRPC server (version %s) on %s:%s | workers=%d",
        SERVER_VERSION, host, port, max_workers
    )

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        log.info("Shutting down...")
        server.stop(0)


if __name__ == "__main__":
    serve()
