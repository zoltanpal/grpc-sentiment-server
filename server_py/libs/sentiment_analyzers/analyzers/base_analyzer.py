import threading

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class SentimentAnalyzerSingleton:
    """
    A singleton class for performing sentiment analysis.
    Ensures that the model, tokenizer, and pipeline are initialized only once per model name,
    even in a multi-threaded environment.
    """

    # Stores the instances of the sentiment analyzer per model name
    _instances = {}  # type: ignore
    _lock = threading.Lock()  # Ensures thread safety during initialization

    def __new__(cls, model_name):
        """
        Returns a singleton instance of SentimentAnalyzer for a given model name.
        If the instance does not exist, it initializes the model, tokenizer, and pipeline.

        Args:
            model_name (str): The name of the model to use for sentiment analysis.
        Returns:
            SentimentAnalyzerSingleton: An instance of the sentiment analyzer for the model.
        """
        with cls._lock:
            if model_name not in cls._instances:
                # Create a new instance if it does not exist
                instance = super().__new__(cls)
                instance._init_model(model_name)
                cls._instances[model_name] = instance
            return cls._instances[model_name]

    def _init_model(self, model_name):
        """
        Initializes the model, tokenizer, and sentiment analysis pipeline.
        This method is only called once per model name during the first instance creation.

        Args:
            model_name (str): The name of the model to load from Hugging Face.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            top_k=None,  # Ensures all sentiment labels are returned
        )

    def analyze(self, text: str):
        """
        Analyzes the sentiment of a given text.
        This method uses the initialized pipeline to predict the sentiment of the input text.

        Args:
            text (str): The text to analyze for sentiment.
        Returns:
            list: A list of sentiment predictions, including labels and confidence scores.
        """
        if not text:
            raise ValueError("Missing text to analyze")

        # Run sentiment analysis using the pipeline
        return self.pipeline(text)
