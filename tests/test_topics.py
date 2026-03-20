"""Tests for the topic extraction module."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.topics import (
    RETAIL_SEED_TOPICS,
    TOPIC_LABELS,
    TopicResult,
    keyword_topic_classify,
)


class TestKeywordTopicClassify:
    def test_sizing_topic(self):
        result = keyword_topic_classify("This hoodie runs small, the sizing is completely off")
        assert result.topic_label == "sizing_and_fit"
        assert result.confidence > 0
        assert len(result.keywords) > 0

    def test_quality_topic(self):
        result = keyword_topic_classify("The stitching is terrible and the material is cheap")
        assert result.topic_label == "quality"

    def test_delivery_topic(self):
        result = keyword_topic_classify("My order was late and the packaging was damaged in transit")
        assert result.topic_label == "delivery"

    def test_customer_service_topic(self):
        result = keyword_topic_classify("The returns process took forever and I got no refund")
        assert result.topic_label == "customer_service"

    def test_style_topic(self):
        result = keyword_topic_classify("The colour looks different to the photo on the website")
        assert result.topic_label == "style_and_design"

    def test_price_topic(self):
        result = keyword_topic_classify("This is so overpriced it's not worth it at all")
        assert result.topic_label == "price_and_value"

    def test_no_topic_match(self):
        result = keyword_topic_classify("I ordered something yesterday")
        assert result.topic_label == "other"
        assert result.topic_id == -1
        assert result.confidence == 0.0

    def test_returns_topic_result_type(self):
        result = keyword_topic_classify("Sizing off")
        assert isinstance(result, TopicResult)
        assert isinstance(result.topic_id, int)
        assert isinstance(result.topic_label, str)
        assert isinstance(result.keywords, list)
        assert isinstance(result.confidence, float)

    def test_multiple_topics_picks_strongest(self):
        text = "The stitching is poor quality and the material is terrible and cheap and falling apart"
        result = keyword_topic_classify(text)
        # Should pick quality since more keywords match
        assert result.topic_label == "quality"

    def test_confidence_bounded(self):
        result = keyword_topic_classify("runs small too big sizing off tight loose fit true to size")
        assert 0 <= result.confidence <= 1.0


class TestSeedTopics:
    def test_all_topics_have_keywords(self):
        for label, keywords in RETAIL_SEED_TOPICS.items():
            assert len(keywords) > 0, f"Topic '{label}' has no seed keywords"

    def test_topic_labels_match_dict_keys(self):
        assert TOPIC_LABELS == list(RETAIL_SEED_TOPICS.keys())

    def test_six_topic_categories(self):
        expected = {
            "sizing_and_fit", "quality", "delivery",
            "customer_service", "style_and_design", "price_and_value",
        }
        assert set(RETAIL_SEED_TOPICS.keys()) == expected
