"""Tests for the returns analysis module."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.returns_analysis import (
    QualityIssue,
    ReturnAlert,
    ReturnsAnalysisResult,
    analyze_returns,
    compute_return_rates,
    extract_quality_issues,
    flag_high_return_skus,
    flag_size_issue_skus,
    return_reason_breakdown,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_returns_df():
    """Small synthetic returns DataFrame for testing."""
    records = []
    # SKU-001: 10 returns, 6 are size_issue → 60% size issue rate
    for i in range(10):
        reason = "size_issue" if i < 6 else "quality"
        records.append({
            "mention_id": f"MNT-{i:04d}",
            "timestamp": pd.Timestamp("2025-10-01") + pd.Timedelta(days=i),
            "platform": "returns_feedback_form",
            "text": "Returning because it runs small" if reason == "size_issue"
                    else "Stitching is coming apart",
            "product_category": "apparel",
            "product_sku": "SKU-001",
            "star_rating": None,
            "author_follower_count": None,
            "verified_purchase": None,
            "order_value_band": "mid",
            "region": "UK",
            "return_reason": reason,
        })

    # SKU-002: 5 returns, no size issues
    for i in range(5):
        records.append({
            "mention_id": f"MNT-{10+i:04d}",
            "timestamp": pd.Timestamp("2025-10-01") + pd.Timedelta(days=i),
            "platform": "returns_feedback_form",
            "text": "Changed my mind about this",
            "product_category": "beauty",
            "product_sku": "SKU-002",
            "star_rating": None,
            "author_follower_count": None,
            "verified_purchase": None,
            "order_value_band": "low",
            "region": "US",
            "return_reason": "changed_mind",
        })

    # Non-return mentions
    for i in range(10):
        records.append({
            "mention_id": f"MNT-{15+i:04d}",
            "timestamp": pd.Timestamp("2025-10-01") + pd.Timedelta(days=i),
            "platform": "trustpilot_review",
            "text": "Great product, love it!",
            "product_category": "apparel",
            "product_sku": "SKU-001",
            "star_rating": 5,
            "author_follower_count": None,
            "verified_purchase": True,
            "order_value_band": "mid",
            "region": "UK",
            "return_reason": None,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Quality issue extraction
# ---------------------------------------------------------------------------

class TestQualityIssueExtraction:
    def test_stitching_defect(self):
        issues = extract_quality_issues("The stitching is coming apart everywhere")
        types = [i.issue_type for i in issues]
        assert "stitching_defect" in types

    def test_pilling(self):
        issues = extract_quality_issues("The material started pilling after one wear")
        types = [i.issue_type for i in issues]
        assert "pilling" in types

    def test_colour_fade(self):
        issues = extract_quality_issues("The colour faded after the first wash cycle")
        types = [i.issue_type for i in issues]
        assert "colour_fade" in types

    def test_zip_defect(self):
        issues = extract_quality_issues("The zipper broke on day one")
        types = [i.issue_type for i in issues]
        assert "zip_button_defect" in types

    def test_damage_on_arrival(self):
        issues = extract_quality_issues("Product arrived damaged with a rip")
        types = [i.issue_type for i in issues]
        assert "damage_on_arrival" in types

    def test_no_issues_found(self):
        issues = extract_quality_issues("I changed my mind about the purchase")
        assert len(issues) == 0

    def test_mention_id_passed_through(self):
        issues = extract_quality_issues("Stitching is bad", mention_id="MNT-001")
        assert issues[0].mention_id == "MNT-001"

    def test_result_type(self):
        issues = extract_quality_issues("The seams are falling apart")
        for issue in issues:
            assert isinstance(issue, QualityIssue)


# ---------------------------------------------------------------------------
# Return rates
# ---------------------------------------------------------------------------

class TestReturnRates:
    def test_compute_rates(self, sample_returns_df):
        rates = compute_return_rates(sample_returns_df)
        assert "product_sku" in rates.columns
        assert "return_rate" in rates.columns
        assert len(rates) > 0

    def test_empty_df(self):
        empty = pd.DataFrame(columns=[
            "platform", "product_sku", "return_reason",
        ])
        rates = compute_return_rates(empty)
        assert len(rates) == 0


# ---------------------------------------------------------------------------
# SKU flagging
# ---------------------------------------------------------------------------

class TestFlagHighReturnSKUs:
    def test_flags_high_rate(self, sample_returns_df):
        rates = compute_return_rates(sample_returns_df)
        # Force a high rate for testing
        rates.loc[rates["product_sku"] == "SKU-001", "return_rate"] = 25.0
        alerts = flag_high_return_skus(rates, return_rate_threshold=15.0)
        skus = [a.sku for a in alerts]
        assert "SKU-001" in skus

    def test_no_flags_below_threshold(self, sample_returns_df):
        rates = compute_return_rates(sample_returns_df)
        rates["return_rate"] = 5.0
        alerts = flag_high_return_skus(rates, return_rate_threshold=15.0)
        assert len(alerts) == 0


class TestFlagSizeIssues:
    def test_flags_size_issue_sku(self, sample_returns_df):
        skus, alerts = flag_size_issue_skus(sample_returns_df, size_issue_threshold=30.0)
        # SKU-001 has 60% size issues
        assert "SKU-001" in skus

    def test_does_not_flag_non_size_sku(self, sample_returns_df):
        skus, alerts = flag_size_issue_skus(sample_returns_df, size_issue_threshold=30.0)
        assert "SKU-002" not in skus


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

class TestAnalyzeReturns:
    def test_full_analysis(self, sample_returns_df):
        result = analyze_returns(sample_returns_df)
        assert isinstance(result, ReturnsAnalysisResult)
        assert not result.return_reason_breakdown.empty
        assert len(result.alerts) > 0
        assert len(result.recommendations) > 0

    def test_size_issue_skus_populated(self, sample_returns_df):
        result = analyze_returns(sample_returns_df)
        assert "SKU-001" in result.size_issue_skus
