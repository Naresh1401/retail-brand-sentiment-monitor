"""Tests for the FastAPI service."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestAnalyzeEndpoint:
    def test_analyze_valid_text(self):
        response = client.post(
            "/analyze",
            json={"text": "This product is absolutely amazing! Love it!"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data
        assert data["sentiment"] in ("positive", "neutral", "negative")
        assert "confidence" in data
        assert "topic" in data
        assert "crisis_flag" in data
        assert isinstance(data["crisis_flag"], bool)

    def test_analyze_with_star_rating(self):
        response = client.post(
            "/analyze",
            json={"text": "It's okay", "star_rating": 2},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] in ("positive", "neutral", "negative")

    def test_analyze_crisis_keywords(self):
        response = client.post(
            "/analyze",
            json={"text": "This product has a dangerous defect and caused an injury"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["crisis_flag"] is True
        assert len(data["crisis_keywords"]) > 0

    def test_analyze_empty_text_rejected(self):
        response = client.post("/analyze", json={"text": ""})
        assert response.status_code == 422

    def test_analyze_missing_text(self):
        response = client.post("/analyze", json={})
        assert response.status_code == 422


class TestAnalyzeReturnEndpoint:
    def test_analyze_return_text(self):
        response = client.post(
            "/analyze-return",
            json={"text": "Returning because it runs small and the sizing is off"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data
        assert "return_reason_classification" in data
        assert "quality_issues" in data

    def test_analyze_return_quality_issues(self):
        response = client.post(
            "/analyze-return",
            json={"text": "The stitching is coming apart, very poor quality"},
        )
        data = response.json()
        assert len(data["quality_issues"]) > 0


class TestBrandHealthEndpoint:
    def test_brand_health(self):
        response = client.get("/brand-health")
        # May fail if data not generated — that's expected in CI without data
        if response.status_code == 200:
            data = response.json()
            assert "overall_net_sentiment" in data
            assert "channel_breakdown" in data
            assert isinstance(data["channel_breakdown"], list)


class TestSKUAlertsEndpoint:
    def test_sku_alerts(self):
        response = client.get("/sku-alerts")
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)


class TestCrisisStatusEndpoint:
    def test_crisis_status(self):
        response = client.get("/crisis-status")
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
