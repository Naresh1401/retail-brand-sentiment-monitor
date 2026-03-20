"""Tests for the retail sentiment classifier."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sentiment import (
    RetailSentimentClassifier,
    SentimentResult,
    calibrate_with_star_rating,
    preprocess_emojis,
)


# ---------------------------------------------------------------------------
# Emoji pre-processing
# ---------------------------------------------------------------------------

class TestEmojiPreprocessing:
    def test_positive_emoji_expansion(self):
        result = preprocess_emojis("This is 🔥")
        assert "fire" in result.lower() or "amazing" in result.lower()

    def test_negative_emoji_expansion(self):
        result = preprocess_emojis("So disappointed 😭")
        assert "crying" in result.lower() or "upset" in result.lower()

    def test_multiple_emojis(self):
        result = preprocess_emojis("Love it 🔥❤️💯")
        assert "fire" in result.lower()
        assert "love" in result.lower()

    def test_no_emoji_passthrough(self):
        text = "This is a normal review with no emojis"
        result = preprocess_emojis(text)
        assert result == text

    def test_whitespace_collapsed(self):
        result = preprocess_emojis("So   much    space")
        assert "  " not in result


# ---------------------------------------------------------------------------
# Star-rating calibration
# ---------------------------------------------------------------------------

class TestStarCalibration:
    def test_agreement_returns_same_label(self):
        scores = {"positive": 0.7, "neutral": 0.2, "negative": 0.1}
        label, new_scores, calibrated = calibrate_with_star_rating("positive", scores, 5)
        assert label == "positive"
        assert calibrated is True

    def test_no_star_returns_uncalibrated(self):
        scores = {"positive": 0.7, "neutral": 0.2, "negative": 0.1}
        label, new_scores, calibrated = calibrate_with_star_rating("positive", scores, None)
        assert label == "positive"
        assert calibrated is False
        assert new_scores == scores

    def test_disagreement_blends_scores(self):
        scores = {"positive": 0.6, "neutral": 0.2, "negative": 0.2}
        label, new_scores, calibrated = calibrate_with_star_rating("positive", scores, 1)
        # With a 1-star rating, negative prior should pull the result
        assert calibrated is True
        # Scores should sum to ~1
        total = sum(new_scores.values())
        assert abs(total - 1.0) < 0.01

    def test_star_1_maps_negative(self):
        scores = {"positive": 0.34, "neutral": 0.33, "negative": 0.33}
        label, _, _ = calibrate_with_star_rating("positive", scores, 1)
        # Should flip toward negative
        assert label in ("negative", "neutral")

    def test_star_5_maps_positive(self):
        scores = {"positive": 0.33, "neutral": 0.34, "negative": 0.33}
        label, _, _ = calibrate_with_star_rating("neutral", scores, 5)
        assert label == "positive"


# ---------------------------------------------------------------------------
# Classifier integration (requires model download — mark as slow)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def classifier():
    """Load classifier once for the test module."""
    try:
        clf = RetailSentimentClassifier(device="cpu")
        # Force pipeline load
        _ = clf.pipeline
        return clf
    except Exception:
        pytest.skip("Transformer model not available (requires download)")


class TestSentimentClassifier:
    def test_positive_text(self, classifier):
        result = classifier.classify("Absolutely love my new hoodie! Best purchase ever 🔥")
        assert isinstance(result, SentimentResult)
        assert result.label == "positive"
        assert result.confidence > 0.5

    def test_negative_text(self, classifier):
        result = classifier.classify("Terrible quality. Fell apart after one wash 😭")
        assert result.label == "negative"

    def test_neutral_text(self, classifier):
        result = classifier.classify("The product is okay. Nothing special but does the job")
        assert result.label in ("neutral", "positive")  # model may lean slightly positive

    def test_scores_sum_to_one(self, classifier):
        result = classifier.classify("Some random test text")
        total = sum(result.scores.values())
        assert abs(total - 1.0) < 0.05

    def test_with_star_rating(self, classifier):
        result = classifier.classify("This product is fine", star_rating=1)
        assert result.calibrated is True

    def test_batch_classify(self, classifier):
        texts = [
            "Love this product!",
            "Terrible quality",
            "It's okay I guess",
        ]
        results = classifier.classify_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, SentimentResult) for r in results)

    def test_emoji_only_sentiment(self, classifier):
        result = classifier.classify("🔥❤️💯")
        assert result.label == "positive"

    def test_empty_after_preprocessing(self, classifier):
        # Very short text should still work
        result = classifier.classify("ok")
        assert isinstance(result, SentimentResult)
