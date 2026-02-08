# libs/sentiment_analyzers/analyzers/base_analyzer.py
import threading
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from libs.sentiment_analyzers.models.sentiments import Sentiments  # type: ignore
from libs.sentiment_analyzers.models.sentiments import LABEL_MAPPING_ROBERTA  # type: ignore

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

    def _map_predictions_to_sentiments(self, predictions: list, label_mapping: dict) -> Sentiments:
        """
        Common logic to convert model predictions to Sentiments dataclass.
        
        Args:
            predictions: List of dicts with 'label' and 'score' keys
            label_mapping: Dict mapping raw labels to sentiment keys
            
        Returns:
            Sentiments object with mapped scores
        """
        scores = {}
        for item in predictions:
            raw_label = item["label"]
            label = label_mapping.get(raw_label, raw_label)
            scores[label] = round(float(item["score"]), 4)
        return Sentiments(**scores)

    def _map_batch_predictions_to_sentiments(self, batch_predictions: list, label_mapping: dict) -> list:
        """
        Common logic to convert batch predictions to Sentiments objects.
        
        Args:
            batch_predictions: List of prediction lists
            label_mapping: Dict mapping raw labels to sentiment keys
            
        Returns:
            List of Sentiments objects
        """
        results = []
        for predictions in batch_predictions:
            results.append(self._map_predictions_to_sentiments(predictions, label_mapping))
        return results
