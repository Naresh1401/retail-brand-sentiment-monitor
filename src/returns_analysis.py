"""
Returns analysis module.

Analyses returns feedback to surface:
- Sentiment on return reasons
- High-return-rate SKU flagging
- Size-issue hotspot detection
- NLP extraction of specific quality defect language
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections import Counter

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Quality issue keyword lexicon
# ---------------------------------------------------------------------------
QUALITY_ISSUE_PATTERNS: dict[str, list[str]] = {
    "stitching_defect": [
        r"stitch(?:ing)?", r"seam[s]?", r"coming apart", r"unravell?(?:ing|ed)",
        r"thread[s]? (?:loose|pulling|coming)",
    ],
    "material_quality": [
        r"thin material", r"see[- ]?through", r"cheap feel",
        r"fabric quality", r"material feels? cheap",
    ],
    "pilling": [
        r"pill(?:ing|ed|s)", r"bobb(?:ling|ly)", r"fuzz(?:y|ing)",
    ],
    "colour_fade": [
        r"colour? fad(?:ed|ing)", r"color fad(?:ed|ing)", r"dye (?:ran|running|bleed)",
        r"wash(?:ed)? out",
    ],
    "zip_button_defect": [
        r"zip(?:per)? (?:broke|broken|stuck|jammed)",
        r"button (?:fell|missing|loose|came off)",
    ],
    "sizing_inconsistency": [
        r"runs? (?:small|large|big|tiny)", r"sizing? (?:off|wrong|incorrect)",
        r"size (?:chart|guide) (?:wrong|incorrect|inaccurate)",
    ],
    "damage_on_arrival": [
        r"arrived? damaged", r"came (?:ripped|torn|damaged|broken)",
        r"damaged in (?:transit|shipping|delivery)", r"hole in",
    ],
}


@dataclass
class ReturnAlert:
    """Alert raised for a problematic SKU."""
    sku: str
    alert_type: str  # high_return_rate / size_issue_hotspot
    metric_value: float  # e.g. return rate %
    threshold: float
    recommendation: str
    product_category: str | None = None


@dataclass
class QualityIssue:
    """A specific quality issue extracted from text."""
    issue_type: str
    matched_text: str
    mention_id: str | None = None


@dataclass
class ReturnsAnalysisResult:
    """Full output of returns analysis."""
    return_reason_breakdown: pd.DataFrame
    sku_return_rates: pd.DataFrame
    alerts: list[ReturnAlert]
    quality_issues: list[QualityIssue]
    size_issue_skus: list[str]
    recommendations: list[str]


def extract_quality_issues(
    text: str,
    mention_id: str | None = None,
) -> list[QualityIssue]:
    """
    Extract specific quality issues from returns feedback text using
    retail-specific regex patterns.
    """
    issues: list[QualityIssue] = []
    text_lower = text.lower()

    for issue_type, patterns in QUALITY_ISSUE_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                issues.append(
                    QualityIssue(
                        issue_type=issue_type,
                        matched_text=match.group(),
                        mention_id=mention_id,
                    )
                )
                break  # one match per issue type per text

    return issues


def compute_return_rates(
    df: pd.DataFrame,
    total_orders_per_sku: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Compute return rates per SKU.

    If total_orders_per_sku is not provided, estimates from the data
    (assumes returns_feedback_form entries represent ~12% of total orders).
    """
    returns_df = df[df["platform"] == "returns_feedback_form"].copy()

    if returns_df.empty:
        return pd.DataFrame(columns=["product_sku", "return_count", "estimated_total_orders", "return_rate"])

    sku_return_counts = returns_df.groupby("product_sku").size().reset_index(name="return_count")

    if total_orders_per_sku:
        sku_return_counts["estimated_total_orders"] = sku_return_counts["product_sku"].map(
            total_orders_per_sku
        )
    else:
        # Estimate: return mentions ≈ 12% of orders → total_orders = return_count / 0.12
        sku_return_counts["estimated_total_orders"] = (
            sku_return_counts["return_count"] / 0.12
        ).astype(int).clip(lower=1)

    sku_return_counts["return_rate"] = (
        sku_return_counts["return_count"]
        / sku_return_counts["estimated_total_orders"]
        * 100
    ).round(2)

    return sku_return_counts.sort_values("return_rate", ascending=False).reset_index(drop=True)


def flag_high_return_skus(
    sku_rates: pd.DataFrame,
    return_rate_threshold: float = 15.0,
) -> list[ReturnAlert]:
    """Flag SKUs with return rate above threshold."""
    alerts = []
    high = sku_rates[sku_rates["return_rate"] > return_rate_threshold]

    for _, row in high.iterrows():
        alerts.append(
            ReturnAlert(
                sku=row["product_sku"],
                alert_type="high_return_rate",
                metric_value=row["return_rate"],
                threshold=return_rate_threshold,
                recommendation=(
                    f"Investigate quality and listing accuracy for {row['product_sku']}. "
                    f"Return rate {row['return_rate']:.1f}% exceeds {return_rate_threshold}% threshold."
                ),
            )
        )

    return alerts


def flag_size_issue_skus(
    df: pd.DataFrame,
    size_issue_threshold: float = 30.0,
) -> tuple[list[str], list[ReturnAlert]]:
    """
    Auto-flag SKUs where >30% of returns cite 'size_issue'.
    """
    returns_df = df[df["platform"] == "returns_feedback_form"].copy()
    if returns_df.empty:
        return [], []

    sku_groups = returns_df.groupby("product_sku")
    flagged_skus = []
    alerts = []

    for sku, group in sku_groups:
        total_returns = len(group)
        size_returns = (group["return_reason"] == "size_issue").sum()
        pct = (size_returns / total_returns * 100) if total_returns > 0 else 0

        if pct > size_issue_threshold:
            flagged_skus.append(sku)
            alerts.append(
                ReturnAlert(
                    sku=sku,
                    alert_type="size_issue_hotspot",
                    metric_value=round(pct, 1),
                    threshold=size_issue_threshold,
                    recommendation=f"Update size guide for {sku}. {pct:.0f}% of returns cite sizing issues.",
                )
            )

    return flagged_skus, alerts


def return_reason_breakdown(
    df: pd.DataFrame,
    group_by: str = "product_category",
) -> pd.DataFrame:
    """
    Breakdown of return reasons by category or SKU.
    """
    returns_df = df[df["platform"] == "returns_feedback_form"].copy()
    if returns_df.empty:
        return pd.DataFrame()

    breakdown = (
        returns_df.groupby([group_by, "return_reason"])
        .size()
        .reset_index(name="count")
    )
    total_by_group = returns_df.groupby(group_by).size().reset_index(name="total")
    breakdown = breakdown.merge(total_by_group, on=group_by)
    breakdown["percentage"] = (breakdown["count"] / breakdown["total"] * 100).round(1)

    return breakdown.sort_values([group_by, "count"], ascending=[True, False])


def analyze_returns(
    df: pd.DataFrame,
    return_rate_threshold: float = 15.0,
    size_issue_threshold: float = 30.0,
) -> ReturnsAnalysisResult:
    """
    Full returns analysis pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Full mentions DataFrame.
    return_rate_threshold : float
        Percentage threshold for high return rate alerts.
    size_issue_threshold : float
        Percentage threshold for size issue flagging.

    Returns
    -------
    ReturnsAnalysisResult
    """
    returns_df = df[df["platform"] == "returns_feedback_form"].copy()

    # 1) Reason breakdown
    reason_df = return_reason_breakdown(df, group_by="product_category")

    # 2) SKU return rates
    sku_rates = compute_return_rates(df)

    # 3) High return rate flags
    rate_alerts = flag_high_return_skus(sku_rates, return_rate_threshold)

    # 4) Size issue flags
    size_skus, size_alerts = flag_size_issue_skus(df, size_issue_threshold)

    all_alerts = rate_alerts + size_alerts

    # 5) NLP quality issue extraction from returns text
    quality_issues: list[QualityIssue] = []
    for _, row in returns_df.iterrows():
        issues = extract_quality_issues(
            row["text"],
            mention_id=row.get("mention_id"),
        )
        quality_issues.extend(issues)

    # 6) Recommendations
    recommendations: list[str] = []
    for alert in all_alerts:
        recommendations.append(alert.recommendation)

    # Top quality issues
    issue_counts = Counter(qi.issue_type for qi in quality_issues)
    for issue_type, count in issue_counts.most_common(3):
        recommendations.append(
            f"Quality issue '{issue_type}' mentioned in {count} return feedback entries. "
            f"Investigate root cause with product team."
        )

    return ReturnsAnalysisResult(
        return_reason_breakdown=reason_df,
        sku_return_rates=sku_rates,
        alerts=all_alerts,
        quality_issues=quality_issues,
        size_issue_skus=size_skus,
        recommendations=recommendations,
    )
