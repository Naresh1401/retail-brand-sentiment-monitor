"""
Retail-tuned sentiment classifier.

Uses cardiffnlp/twitter-roberta-base-sentiment-latest with emoji
pre-processing and star-rating calibration for the retail domain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# ---------------------------------------------------------------------------
# Emoji mapping — decoded before the text hits the transformer
# ---------------------------------------------------------------------------
POSITIVE_EMOJIS = {
    "🔥": " fire amazing ",
    "❤️": " love ",
    "💯": " perfect hundred ",
    "😍": " love heart eyes ",
    "🙏": " thankful blessed ",
    "🙌": " celebration ",
    "👏": " applause great ",
    "✨": " sparkle amazing ",
    "💪": " strong great ",
    "👍": " thumbs up good ",
    "🥰": " love adore ",
    "😊": " happy satisfied ",
    "💕": " love hearts ",
    "👨‍🍳": " perfect chef kiss ",
    "🎉": " celebration ",
}

NEGATIVE_EMOJIS = {
    "😭": " crying upset ",
    "💔": " heartbroken disappointed ",
    "🙅": " no refuse reject ",
    "😤": " angry frustrated ",
    "🤬": " furious angry ",
    "👎": " thumbs down bad ",
    "😡": " angry rage ",
    "🤡": " clown ridiculous ",
    "💀": " dead awful ",
    "🙃": " sarcastic ironic ",
    "😒": " unimpressed ",
    "🚫": " no stop ",
}

ALL_EMOJIS = {**POSITIVE_EMOJIS, **NEGATIVE_EMOJIS}

# Model label mapping for cardiffnlp/twitter-roberta-base-sentiment-latest
# The model outputs: LABEL_0 = negative, LABEL_1 = neutral, LABEL_2 = positive
LABEL_MAP = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}

# Retail domain threshold adjustments
# Retail text tends to read more neutral than social media, so we lower the
# positive threshold and raise the negative threshold slightly
RETAIL_THRESHOLDS = {
    "positive_min_confidence": 0.35,
    "negative_min_confidence": 0.35,
    "neutral_fallback": True,
}


@dataclass
class SentimentResult:
    """Output of the sentiment classifier."""

    label: str  # positive / neutral / negative
    confidence: float  # 0-1
    scores: dict[str, float]  # all three class probabilities
    calibrated: bool  # whether star-rating calibration was applied


def preprocess_emojis(text: str) -> str:
    """Replace common retail emojis with sentiment-bearing text tokens."""
    for emoji_char, replacement in ALL_EMOJIS.items():
        text = text.replace(emoji_char, replacement)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calibrate_with_star_rating(
    model_label: str,
    model_scores: dict[str, float],
    star_rating: int | None,
) -> tuple[str, dict[str, float], bool]:
    """
    Calibrate model output with star rating when available.

    Star-rating mapping:
        1-2 stars → negative
        3 stars   → neutral
        4-5 stars → positive

    If model and star rating disagree, blend the scores.
    """
    if star_rating is None:
        return model_label, model_scores, False

    star_sentiment = (
        "negative" if star_rating <= 2
        else "neutral" if star_rating == 3
        else "positive"
    )

    if star_sentiment == model_label:
        return model_label, model_scores, True

    # Disagreement: blend model scores with star-based prior (60/40 model/star)
    star_prior = {
        "positive": 0.0,
        "neutral": 0.0,
        "negative": 0.0,
    }
    star_prior[star_sentiment] = 1.0

    blended = {}
    for key in model_scores:
        blended[key] = 0.6 * model_scores[key] + 0.4 * star_prior.get(key, 0.0)

    # Normalise
    total = sum(blended.values())
    blended = {k: v / total for k, v in blended.items()}
    final_label = max(blended, key=blended.get)

    return final_label, blended, True


class RetailSentimentClassifier:
    """
    Sentiment classifier optimised for retail brand monitoring.

    Wraps cardiffnlp/twitter-roberta-base-sentiment-latest with:
    - Emoji pre-processing
    - Retail threshold tuning
    - Star-rating calibration
    """

    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    def __init__(self, device: str = "cpu", batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self._pipeline = None

    @property
    def pipeline(self):
        """Lazy-load the transformer pipeline."""
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.MODEL_NAME,
                tokenizer=self.MODEL_NAME,
                device=self.device if self.device == "cpu" else int(self.device.split(":")[-1]),
                top_k=None,  # return all scores
                max_length=512,
                truncation=True,
            )
        return self._pipeline

    def classify(
        self,
        text: str,
        star_rating: int | None = None,
    ) -> SentimentResult:
        """
        Classify a single text.

        Parameters
        ----------
        text : str
            The review / comment / mention text.
        star_rating : int or None
            Optional star rating (1-5) for calibration.

        Returns
        -------
        SentimentResult
        """
        processed_text = preprocess_emojis(text)

        raw_output = self.pipeline(processed_text)[0]

        # Convert list-of-dicts to {label: score}
        scores = {}
        for item in raw_output:
            label = item["label"].lower()
            scores[label] = item["score"]

        model_label = max(scores, key=scores.get)

        # Apply retail threshold: if top confidence is too low, fallback to neutral
        if (
            RETAIL_THRESHOLDS["neutral_fallback"]
            and scores[model_label] < RETAIL_THRESHOLDS.get(
                f"{model_label}_min_confidence", 0.35
            )
        ):
            model_label = "neutral"

        # Calibrate with star rating
        final_label, final_scores, calibrated = calibrate_with_star_rating(
            model_label, scores, star_rating
        )

        return SentimentResult(
            label=final_label,
            confidence=final_scores[final_label],
            scores=final_scores,
            calibrated=calibrated,
        )

    def classify_batch(
        self,
        texts: list[str],
        star_ratings: list[int | None] | None = None,
    ) -> list[SentimentResult]:
        """
        Classify a batch of texts.

        Parameters
        ----------
        texts : list[str]
        star_ratings : list[int|None] or None

        Returns
        -------
        list[SentimentResult]
        """
        if star_ratings is None:
            star_ratings = [None] * len(texts)

        processed = [preprocess_emojis(t) for t in texts]

        results = []
        for i in range(0, len(processed), self.batch_size):
            batch_texts = processed[i : i + self.batch_size]
            batch_stars = star_ratings[i : i + self.batch_size]
            raw_outputs = self.pipeline(batch_texts)

            for raw, stars in zip(raw_outputs, batch_stars):
                scores = {}
                for item in raw:
                    label = item["label"].lower()
                    scores[label] = item["score"]

                model_label = max(scores, key=scores.get)

                if (
                    RETAIL_THRESHOLDS["neutral_fallback"]
                    and scores[model_label]
                    < RETAIL_THRESHOLDS.get(f"{model_label}_min_confidence", 0.35)
                ):
                    model_label = "neutral"

                final_label, final_scores, calibrated = calibrate_with_star_rating(
                    model_label, scores, stars
                )

                results.append(
                    SentimentResult(
                        label=final_label,
                        confidence=final_scores[final_label],
                        scores=final_scores,
                        calibrated=calibrated,
                    )
                )

        return results


# Module-level convenience function
_classifier: RetailSentimentClassifier | None = None


def get_classifier(device: str = "cpu") -> RetailSentimentClassifier:
    """Return a module-level singleton classifier."""
    global _classifier
    if _classifier is None:
        _classifier = RetailSentimentClassifier(device=device)
    return _classifier


def analyze_sentiment(
    text: str,
    star_rating: int | None = None,
    device: str = "cpu",
) -> SentimentResult:
    """Quick single-text analysis."""
    return get_classifier(device).classify(text, star_rating)
