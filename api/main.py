"""
FastAPI service for retail brand sentiment monitoring.

Endpoints:
- POST /analyze         → sentiment + topic + crisis flag for single text
- POST /analyze-return  → return reason + quality issues for return feedback
- GET  /brand-health    → rolling 7-day net sentiment by channel
- GET  /sku-alerts      → SKUs with negative sentiment spike or high return rate
- GET  /crisis-status   → active crisis alerts with severity
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure src is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sentiment import RetailSentimentClassifier, preprocess_emojis
from src.topics import keyword_topic_classify
from src.returns_analysis import extract_quality_issues, analyze_returns
from src.crisis_detector import CrisisDetector, QUALITY_ALERT_KEYWORDS

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Retail Brand Sentiment Monitor",
    description="NLP-powered brand monitoring API for DTC retail",
    version="1.0.0",
)

# Lazy-loaded singletons
_classifier: RetailSentimentClassifier | None = None
_crisis_detector: CrisisDetector | None = None
_data: pd.DataFrame | None = None

DATA_PATH = ROOT / "data" / "retail_mentions.csv"


def get_classifier() -> RetailSentimentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = RetailSentimentClassifier(device="cpu")
    return _classifier


def get_crisis_detector() -> CrisisDetector:
    global _crisis_detector
    if _crisis_detector is None:
        _crisis_detector = CrisisDetector(crisis_log_dir=str(ROOT / "crisis_logs"))
    return _crisis_detector


def get_data() -> pd.DataFrame:
    """Load and cache the mentions dataset with sentiment labels."""
    global _data
    if _data is None:
        if not DATA_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Dataset not generated yet. Run: python data/generate_data.py",
            )
        _data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
        # Add synthetic sentiment labels if not present
        if "sentiment" not in _data.columns:
            _data["sentiment"] = _data.apply(_infer_sentiment_from_star, axis=1)
    return _data


def _infer_sentiment_from_star(row: pd.Series) -> str:
    """Quick heuristic when model labels aren't pre-computed."""
    star = row.get("star_rating")
    if pd.notna(star):
        if star <= 2:
            return "negative"
        elif star == 3:
            return "neutral"
        else:
            return "positive"
    # Keyword heuristic for non-review platforms
    text = str(row.get("text", "")).lower()
    neg_words = ["disappointed", "terrible", "awful", "worst", "poor", "defect", "broke", "waste"]
    pos_words = ["love", "amazing", "perfect", "great", "excellent", "best", "happy", "obsessed"]
    neg_count = sum(1 for w in neg_words if w in text)
    pos_count = sum(1 for w in pos_words if w in text)
    if neg_count > pos_count:
        return "negative"
    elif pos_count > neg_count:
        return "positive"
    return "neutral"


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Review or mention text")
    star_rating: int | None = Field(None, ge=1, le=5, description="Optional star rating 1-5")
    platform: str | None = Field(None, description="Source platform")
    product_sku: str | None = Field(None, description="Product SKU if known")


class AnalyzeResponse(BaseModel):
    sentiment: str
    confidence: float
    scores: dict[str, float]
    topic: str
    topic_keywords: list[str]
    crisis_flag: bool
    crisis_keywords: list[str]


class ReturnAnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Returns feedback text")
    product_sku: str | None = None


class ReturnAnalyzeResponse(BaseModel):
    sentiment: str
    confidence: float
    return_reason_classification: str
    quality_issues: list[dict]


class BrandHealthResponse(BaseModel):
    period: str
    overall_net_sentiment: float
    channel_breakdown: list[dict]


class SKUAlertResponse(BaseModel):
    sku: str
    alert_type: str
    metric: str
    value: float
    recommendation: str


class CrisisStatusResponse(BaseModel):
    alert_id: str
    severity: int
    severity_label: str
    alert_type: str
    description: str
    suggested_action: str
    timestamp: str
    resolved: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(req: AnalyzeRequest):
    """Analyze a single mention: sentiment + topic + crisis flag."""
    clf = get_classifier()
    result = clf.classify(req.text, star_rating=req.star_rating)

    topic = keyword_topic_classify(req.text)

    # Check for crisis keywords
    text_lower = req.text.lower()
    crisis_kws = [kw for kw in QUALITY_ALERT_KEYWORDS if kw in text_lower]
    crisis_flag = len(crisis_kws) > 0

    return AnalyzeResponse(
        sentiment=result.label,
        confidence=round(result.confidence, 4),
        scores={k: round(v, 4) for k, v in result.scores.items()},
        topic=topic.topic_label,
        topic_keywords=topic.keywords,
        crisis_flag=crisis_flag,
        crisis_keywords=crisis_kws,
    )


@app.post("/analyze-return", response_model=ReturnAnalyzeResponse)
def analyze_return(req: ReturnAnalyzeRequest):
    """Analyze a returns feedback text: classify return reason + extract quality issues."""
    clf = get_classifier()
    sentiment = clf.classify(req.text)

    # Return reason classification via topic keywords
    topic = keyword_topic_classify(req.text)

    # Map topic to return reason
    topic_to_reason = {
        "sizing_and_fit": "size_issue",
        "quality": "quality",
        "style_and_design": "not_as_described",
        "delivery": "damaged",
        "other": "changed_mind",
        "price_and_value": "changed_mind",
        "customer_service": "changed_mind",
    }
    return_reason = topic_to_reason.get(topic.topic_label, "changed_mind")

    # Extract quality issues
    issues = extract_quality_issues(req.text)
    issues_dicts = [{"issue_type": qi.issue_type, "matched_text": qi.matched_text} for qi in issues]

    return ReturnAnalyzeResponse(
        sentiment=sentiment.label,
        confidence=round(sentiment.confidence, 4),
        return_reason_classification=return_reason,
        quality_issues=issues_dicts,
    )


@app.get("/brand-health", response_model=BrandHealthResponse)
def brand_health():
    """Rolling 7-day net sentiment score by channel."""
    df = get_data()

    cutoff = df["timestamp"].max() - timedelta(days=7)
    recent = df[df["timestamp"] >= cutoff]

    channels = recent.groupby("platform")["sentiment"].value_counts().unstack(fill_value=0)
    channel_list = []
    for platform in channels.index:
        row = channels.loc[platform]
        pos = int(row.get("positive", 0))
        neg = int(row.get("negative", 0))
        neu = int(row.get("neutral", 0))
        total = pos + neg + neu
        net = ((pos - neg) / total * 100) if total > 0 else 0
        channel_list.append({
            "platform": platform,
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "total": total,
            "net_sentiment": round(net, 1),
        })

    # Overall net sentiment
    total_pos = recent[recent["sentiment"] == "positive"].shape[0]
    total_neg = recent[recent["sentiment"] == "negative"].shape[0]
    total_all = len(recent)
    overall = ((total_pos - total_neg) / total_all * 100) if total_all > 0 else 0

    return BrandHealthResponse(
        period=f"{cutoff.date()} to {df['timestamp'].max().date()}",
        overall_net_sentiment=round(overall, 1),
        channel_breakdown=channel_list,
    )


@app.get("/sku-alerts", response_model=list[SKUAlertResponse])
def sku_alerts():
    """SKUs with negative sentiment spike or high return rate."""
    df = get_data()

    alerts: list[SKUAlertResponse] = []

    # Negative sentiment by SKU (last 7 days)
    cutoff = df["timestamp"].max() - timedelta(days=7)
    recent = df[df["timestamp"] >= cutoff]

    sku_sentiment = recent.groupby("product_sku")["sentiment"].value_counts().unstack(fill_value=0)
    for sku in sku_sentiment.index:
        row = sku_sentiment.loc[sku]
        neg = int(row.get("negative", 0))
        total = int(row.sum())
        neg_pct = (neg / total * 100) if total > 0 else 0
        if neg_pct > 50 and total >= 5:
            alerts.append(SKUAlertResponse(
                sku=sku,
                alert_type="negative_sentiment_spike",
                metric="negative_percentage_7d",
                value=round(neg_pct, 1),
                recommendation=f"Investigate {sku}: {neg_pct:.0f}% negative sentiment in last 7 days.",
            ))

    # High return rate SKUs
    returns_df = df[df["platform"] == "returns_feedback_form"]
    sku_returns = returns_df.groupby("product_sku").size().reset_index(name="return_count")
    for _, r in sku_returns.iterrows():
        estimated_orders = r["return_count"] / 0.12
        rate = (r["return_count"] / estimated_orders * 100) if estimated_orders > 0 else 0
        if rate > 15 and r["return_count"] >= 3:
            alerts.append(SKUAlertResponse(
                sku=r["product_sku"],
                alert_type="high_return_rate",
                metric="estimated_return_rate",
                value=round(rate, 1),
                recommendation=f"Review quality and sizing for {r['product_sku']}.",
            ))

    return alerts


@app.get("/crisis-status", response_model=list[CrisisStatusResponse])
def crisis_status():
    """Active crisis alerts with severity level."""
    df = get_data()
    detector = get_crisis_detector()
    alerts = detector.scan(df, sentiment_col="sentiment")

    return [
        CrisisStatusResponse(
            alert_id=a.alert_id,
            severity=a.severity,
            severity_label=a.severity_label,
            alert_type=a.alert_type,
            description=a.description,
            suggested_action=a.suggested_action,
            timestamp=a.timestamp,
            resolved=a.resolved,
        )
        for a in alerts[:50]  # Cap output
    ]


@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
