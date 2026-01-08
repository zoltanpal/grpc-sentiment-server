import math
from dataclasses import asdict, dataclass, field

# Hungarian sentiment results mapping
LABEL_MAPPING_ROBERTA = {
    "LABEL_0": "very_negative",
    "LABEL_1": "negative",
    "LABEL_2": "neutral",
    "LABEL_3": "positive",
    "LABEL_4": "very_positive",
}


@dataclass
class Sentiments:
    """
    A data class to represent sentiment scores for text analysis.

    Attributes:
        negative (float): The negative sentiment score (default: 0.0).
        neutral (float): The neutral sentiment score (default: 0.0).
        positive (float): The positive sentiment score (default: 0.0).
        compound (float): The compound sentiment score, representing the
                          overall polarity (default: 0.0). If not provided,
                          it is calculated based on `positive` and `negative`.
    """

    negative: float = field(default=0.0)
    very_negative: float = field(default=0.0)
    neutral: float = field(default=0.0)
    positive: float = field(default=0.0)
    very_positive: float = field(default=0.0)
    compound: float = field(default=0.0)

    def asdict(self) -> dict:
        """
        Converts the Sentiments object to a dictionary.

        Returns:
            dict: A dictionary representation of the sentiment scores.
        """
        return asdict(self)

    def calculate_compound(self) -> float:
        """
        Calculates the compound sentiment score based on `positive` and `negative`.

        The compound score is a normalized value in the range [-1, 1] using
        the hyperbolic tangent (tanh) function.

        Returns:
            float: The calculated compound sentiment score.
        """
        score = self.positive - self.negative
        return round(math.tanh(score), 4)

    def __post_init__(self):
        """
        Ensures the compound score is calculated if not explicitly set.

        If the compound score is left as its default value (0.0), it is
        recalculated based on the `positive` and `negative` scores.
        """

        # Round all float fields to 4 decimals
        self.negative = round(self.negative, 4)
        self.very_negative = round(self.very_negative, 4)
        self.neutral = round(self.neutral, 4)
        self.positive = round(self.positive, 4)
        self.very_positive = round(self.very_positive, 4)

        if self.compound == 0.0:
            self.compound = self.calculate_compound()
        else:
            self.compound = round(self.compound, 4)
