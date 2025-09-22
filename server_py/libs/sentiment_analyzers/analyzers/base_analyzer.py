# libs/sentiment_analyzers/analyzers/base_analyzer.py
import threading
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

class SentimentAnalyzerSingleton:
    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        # Ensure subclasses define model_name
        if not getattr(cls, "model_name", None):
            raise ValueError(f"{cls.__name__} must define a class attribute 'model_name'")

        with cls._lock:
            if cls not in cls._instances:
                tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
                model = AutoModelForSequenceClassification.from_pretrained(cls.model_name)

                torch.set_num_threads(max(torch.get_num_threads(), 1))
                device = -1  # CPU only

                pipe = pipeline(
                    task="text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    top_k=getattr(cls, "top_k", None),
                    truncation=getattr(cls, "truncation", True),
                    return_all_scores=True,
                )

                inst = super().__new__(cls)
                inst.pipeline = pipe

                # Optional warm-up
                try:
                    inst.pipeline("ok")
                except Exception:
                    pass

                cls._instances[cls] = inst

            return cls._instances[cls]

    def analyze(self, text: str):
        if not text:
            raise ValueError("Missing text to analyze")
        return self.pipeline(text)
