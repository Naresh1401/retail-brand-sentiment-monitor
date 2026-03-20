"""Tests for the crisis detection engine."""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.crisis_detector import (
    CrisisAlert,
    CrisisDetector,
    QUALITY_ALERT_KEYWORDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector(tmp_path):
    """CrisisDetector with temp log dir."""
    return CrisisDetector(crisis_log_dir=str(tmp_path / "crisis_logs"))


@pytest.fixture
def base_mention():
    """Factory for creating mention rows."""
    def _make(
        text="Normal review text",
        sentiment="neutral",
        platform="trustpilot_review",
        sku="SKU-001",
        followers=None,
        ts=None,
    ):
        return {
            "mention_id": "MNT-TEST",
            "timestamp": ts or datetime(2025, 10, 15, 12, 0),
            "platform": platform,
            "text": text,
            "product_category": "apparel",
            "product_sku": sku,
            "star_rating": None,
            "author_follower_count": followers,
            "verified_purchase": None,
            "order_value_band": "mid",
            "region": "UK",
            "return_reason": None,
            "sentiment": sentiment,
        }
    return _make


# ---------------------------------------------------------------------------
# Quality keyword alerts
# ---------------------------------------------------------------------------

class TestQualityKeywordAlerts:
    def test_detects_defect_keyword(self, detector, base_mention):
        df = pd.DataFrame([base_mention(text="This product has a serious defect")])
        alerts = detector.detect_quality_keyword_alerts(df)
        assert len(alerts) > 0
        assert alerts[0].alert_type == "quality_keyword_alert"
        assert alerts[0].severity == 1

    def test_detects_dangerous_keyword(self, detector, base_mention):
        df = pd.DataFrame([base_mention(text="This product is dangerous to use")])
        alerts = detector.detect_quality_keyword_alerts(df)
        assert len(alerts) > 0

    def test_detects_recall_keyword(self, detector, base_mention):
        df = pd.DataFrame([base_mention(text="They should recall this product immediately")])
        alerts = detector.detect_quality_keyword_alerts(df)
        assert len(alerts) > 0

    def test_no_alert_for_normal_text(self, detector, base_mention):
        df = pd.DataFrame([base_mention(text="This hoodie is really comfortable and nice")])
        alerts = detector.detect_quality_keyword_alerts(df)
        assert len(alerts) == 0

    def test_multiple_keywords(self, detector, base_mention):
        df = pd.DataFrame([base_mention(text="This defect is dangerous and toxic")])
        alerts = detector.detect_quality_keyword_alerts(df)
        assert len(alerts) == 1
        # Should find multiple keywords in description
        assert "defect" in alerts[0].description.lower()


# ---------------------------------------------------------------------------
# Viral risk
# ---------------------------------------------------------------------------

class TestViralRisk:
    def test_high_follower_negative(self, detector, base_mention):
        df = pd.DataFrame([
            base_mention(
                text="This product is terrible",
                sentiment="negative",
                platform="tiktok_comment",
                followers=50_000,
            )
        ])
        alerts = detector.detect_viral_risk(df, sentiment_col="sentiment")
        assert len(alerts) > 0
        assert alerts[0].alert_type == "viral_risk"

    def test_no_alert_low_followers(self, detector, base_mention):
        df = pd.DataFrame([
            base_mention(
                text="This is bad",
                sentiment="negative",
                platform="tiktok_comment",
                followers=500,
            )
        ])
        alerts = detector.detect_viral_risk(df, sentiment_col="sentiment")
        assert len(alerts) == 0

    def test_no_alert_positive_high_followers(self, detector, base_mention):
        df = pd.DataFrame([
            base_mention(
                text="Love this!",
                sentiment="positive",
                platform="instagram_comment",
                followers=100_000,
            )
        ])
        alerts = detector.detect_viral_risk(df, sentiment_col="sentiment")
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# Negative volume spike
# ---------------------------------------------------------------------------

class TestNegativeVolumeSpike:
    def test_spike_detected(self, detector, base_mention):
        """Generate >50 negative mentions in a 4-hour window."""
        base_ts = datetime(2025, 10, 15, 12, 0)
        records = []
        for i in range(60):
            records.append(
                base_mention(
                    text="Terrible product",
                    sentiment="negative",
                    ts=base_ts + timedelta(minutes=i * 3),  # 60 in ~3 hours
                )
            )
        df = pd.DataFrame(records)
        alerts = detector.detect_negative_volume_spike(df, sentiment_col="sentiment")
        assert len(alerts) > 0
        assert alerts[0].alert_type == "negative_volume_spike"

    def test_no_spike_low_volume(self, detector, base_mention):
        base_ts = datetime(2025, 10, 15, 12, 0)
        records = [
            base_mention(
                text="Bad product",
                sentiment="negative",
                ts=base_ts + timedelta(minutes=i * 10),
            )
            for i in range(10)
        ]
        df = pd.DataFrame(records)
        alerts = detector.detect_negative_volume_spike(df, sentiment_col="sentiment")
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# Crisis alert structure
# ---------------------------------------------------------------------------

class TestCrisisAlert:
    def test_severity_label(self):
        alert = CrisisAlert(
            alert_id="ALERT-0001",
            timestamp="2025-10-15T12:00:00",
            severity=1,
            alert_type="quality_keyword_alert",
            description="Test alert",
        )
        assert alert.severity_label == "Sev-1 CRITICAL"

    def test_severity_2_label(self):
        alert = CrisisAlert(
            alert_id="ALERT-0002",
            timestamp="2025-10-15T12:00:00",
            severity=2,
            alert_type="viral_risk",
            description="Test",
        )
        assert alert.severity_label == "Sev-2 HIGH"

    def test_severity_3_label(self):
        alert = CrisisAlert(
            alert_id="ALERT-0003",
            timestamp="2025-10-15",
            severity=3,
            alert_type="returns_spike",
            description="Test",
        )
        assert alert.severity_label == "Sev-3 MEDIUM"


class TestFullScan:
    def test_scan_returns_list(self, detector, base_mention):
        df = pd.DataFrame([base_mention()])
        alerts = detector.scan(df, sentiment_col="sentiment")
        assert isinstance(alerts, list)

    def test_resolve_alert(self, detector, base_mention):
        df = pd.DataFrame([base_mention(text="Product has a defect")])
        alerts = detector.scan(df, sentiment_col="sentiment")
        if alerts:
            assert detector.resolve_alert(alerts[0].alert_id) is True
            assert alerts[0].resolved is True

    def test_resolve_nonexistent(self, detector):
        assert detector.resolve_alert("NONEXISTENT") is False

    def test_log_file_created(self, detector, base_mention, tmp_path):
        detector.crisis_log_dir = tmp_path / "logs"
        detector.crisis_log_dir.mkdir()
        df = pd.DataFrame([base_mention(text="This is dangerous and a defect")])
        detector.scan(df, sentiment_col="sentiment")
        log_files = list(detector.crisis_log_dir.glob("*.json"))
        assert len(log_files) > 0
