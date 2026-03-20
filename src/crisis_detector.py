"""
Crisis detection engine for retail brand monitoring.

Monitors for:
1. Negative volume spikes (>50 in 4-hour window)
2. Viral risk (high-follower + negative + product tag)
3. Quality keyword alerts (defect, dangerous, toxic, recall, injury)
4. TikTok growth spikes (>200% WoW)
5. Returns spikes (>5pp increase in 7 days)

Logs alerts to crisis_log and simulates Slack notification.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NEGATIVE_VOLUME_WINDOW_HOURS = 4
NEGATIVE_VOLUME_THRESHOLD = 50  # >50 negative in 4h = 3× normal

VIRAL_FOLLOWER_THRESHOLD = 10_000
VIRAL_PLATFORMS = {"instagram_comment", "tiktok_comment", "twitter_mention", "reddit_post"}

QUALITY_ALERT_KEYWORDS = [
    "defect", "dangerous", "toxic", "recall", "injury",
    "hazard", "unsafe", "poison", "allergic reaction", "chemical",
]

TIKTOK_WOW_GROWTH_THRESHOLD = 2.0  # 200% week-over-week
RETURNS_SPIKE_PP_THRESHOLD = 5.0  # 5 percentage points increase

SEVERITY_MAP = {
    "quality_keyword_alert": 1,
    "viral_risk": 2,
    "negative_volume_spike": 2,
    "tiktok_growth_spike": 2,
    "returns_spike": 3,
}

SUGGESTED_ACTIONS = {
    "quality_keyword_alert": (
        "IMMEDIATE: Escalate to product safety team. "
        "Pause sales of flagged SKU. Draft holding statement. "
        "Contact legal if 'injury' or 'dangerous' mentioned."
    ),
    "viral_risk": (
        "URGENT: Monitor post engagement. Prepare brand response. "
        "Brief social media team for rapid reply within 1 hour. "
        "Consider proactive outreach to the creator."
    ),
    "negative_volume_spike": (
        "URGENT: Investigate root cause — product issue, delivery failure, or PR event? "
        "Brief CS team to expect surge. Prepare FAQ document."
    ),
    "tiktok_growth_spike": (
        "MONITOR: TikTok volume growing rapidly. Check if organic vs coordinated. "
        "Brief content team. Prepare response content if negative."
    ),
    "returns_spike": (
        "INVESTIGATE: SKU return rate spiking. Pull QC reports. "
        "Consider pausing inventory replenishment. Update product listing if needed."
    ),
}


@dataclass
class CrisisAlert:
    """A single crisis alert."""
    alert_id: str
    timestamp: str
    severity: int  # 1 = critical, 2 = high, 3 = medium
    alert_type: str
    description: str
    affected_skus: list[str] = field(default_factory=list)
    affected_platforms: list[str] = field(default_factory=list)
    mention_count: int = 0
    suggested_action: str = ""
    resolved: bool = False

    @property
    def severity_label(self) -> str:
        return {1: "Sev-1 CRITICAL", 2: "Sev-2 HIGH", 3: "Sev-3 MEDIUM"}.get(
            self.severity, "UNKNOWN"
        )


class CrisisDetector:
    """
    Retail brand crisis detection engine.

    Scans a mentions DataFrame for five categories of crisis signals
    and produces alerts with severity ratings and suggested actions.
    """

    def __init__(
        self,
        crisis_log_dir: str = "crisis_logs",
        negative_volume_threshold: int = NEGATIVE_VOLUME_THRESHOLD,
        viral_follower_threshold: int = VIRAL_FOLLOWER_THRESHOLD,
        tiktok_wow_threshold: float = TIKTOK_WOW_GROWTH_THRESHOLD,
        returns_spike_threshold: float = RETURNS_SPIKE_PP_THRESHOLD,
    ):
        self.crisis_log_dir = Path(crisis_log_dir)
        self.crisis_log_dir.mkdir(parents=True, exist_ok=True)
        self.negative_volume_threshold = negative_volume_threshold
        self.viral_follower_threshold = viral_follower_threshold
        self.tiktok_wow_threshold = tiktok_wow_threshold
        self.returns_spike_threshold = returns_spike_threshold

        self._alert_counter = 0
        self.active_alerts: list[CrisisAlert] = []

    def _next_alert_id(self) -> str:
        self._alert_counter += 1
        return f"ALERT-{self._alert_counter:04d}"

    # ----- Detection methods -----

    def detect_negative_volume_spike(
        self,
        df: pd.DataFrame,
        sentiment_col: str = "sentiment",
    ) -> list[CrisisAlert]:
        """
        Detect >50 negative mentions in any 4-hour rolling window.
        """
        alerts = []
        neg = df[df[sentiment_col] == "negative"].copy()
        if neg.empty:
            return alerts

        neg = neg.set_index("timestamp").sort_index()
        window = f"{NEGATIVE_VOLUME_WINDOW_HOURS}h"

        # Rolling count
        neg["_count"] = 1
        rolling = neg["_count"].rolling(window).sum()

        spikes = rolling[rolling > self.negative_volume_threshold]
        if spikes.empty:
            return alerts

        # Group contiguous spikes into distinct events
        spike_times = spikes.index.to_series()
        gaps = spike_times.diff() > pd.Timedelta(hours=NEGATIVE_VOLUME_WINDOW_HOURS)
        event_ids = gaps.cumsum()

        for _, event_group in spike_times.groupby(event_ids):
            peak_time = event_group.iloc[-1]
            window_start = peak_time - pd.Timedelta(hours=NEGATIVE_VOLUME_WINDOW_HOURS)
            window_mentions = neg.loc[window_start:peak_time]

            skus = window_mentions["product_sku"].value_counts().head(3).index.tolist()
            platforms = window_mentions["platform"].value_counts().head(3).index.tolist()

            alert = CrisisAlert(
                alert_id=self._next_alert_id(),
                timestamp=str(peak_time),
                severity=SEVERITY_MAP["negative_volume_spike"],
                alert_type="negative_volume_spike",
                description=(
                    f"Negative mention volume spike: {len(window_mentions)} negative mentions "
                    f"in {NEGATIVE_VOLUME_WINDOW_HOURS}h window ending {peak_time}. "
                    f"Top SKUs: {', '.join(skus)}."
                ),
                affected_skus=skus,
                affected_platforms=platforms,
                mention_count=len(window_mentions),
                suggested_action=SUGGESTED_ACTIONS["negative_volume_spike"],
            )
            alerts.append(alert)

        return alerts

    def detect_viral_risk(
        self,
        df: pd.DataFrame,
        sentiment_col: str = "sentiment",
    ) -> list[CrisisAlert]:
        """
        Detect mentions from high-follower accounts with negative sentiment
        and a product SKU tag.
        """
        alerts = []
        mask = (
            (df[sentiment_col] == "negative")
            & (df["platform"].isin(VIRAL_PLATFORMS))
            & (df["author_follower_count"].fillna(0) > self.viral_follower_threshold)
            & (df["product_sku"].notna())
        )
        viral = df[mask]

        for _, row in viral.iterrows():
            alert = CrisisAlert(
                alert_id=self._next_alert_id(),
                timestamp=str(row["timestamp"]),
                severity=SEVERITY_MAP["viral_risk"],
                alert_type="viral_risk",
                description=(
                    f"Viral risk: {row['platform']} post by user with "
                    f"{int(row['author_follower_count']):,} followers. "
                    f"Negative sentiment about {row['product_sku']}."
                ),
                affected_skus=[row["product_sku"]],
                affected_platforms=[row["platform"]],
                mention_count=1,
                suggested_action=SUGGESTED_ACTIONS["viral_risk"],
            )
            alerts.append(alert)

        return alerts

    def detect_quality_keyword_alerts(
        self,
        df: pd.DataFrame,
    ) -> list[CrisisAlert]:
        """
        Immediate Sev-1 alert for safety/quality keywords.
        """
        alerts = []
        for _, row in df.iterrows():
            text_lower = row["text"].lower()
            matched = [kw for kw in QUALITY_ALERT_KEYWORDS if kw in text_lower]
            if matched:
                alert = CrisisAlert(
                    alert_id=self._next_alert_id(),
                    timestamp=str(row["timestamp"]),
                    severity=SEVERITY_MAP["quality_keyword_alert"],
                    alert_type="quality_keyword_alert",
                    description=(
                        f"Quality/safety keyword detected: {', '.join(matched)}. "
                        f"Platform: {row['platform']}. SKU: {row.get('product_sku', 'N/A')}."
                    ),
                    affected_skus=[row["product_sku"]] if pd.notna(row.get("product_sku")) else [],
                    affected_platforms=[row["platform"]],
                    mention_count=1,
                    suggested_action=SUGGESTED_ACTIONS["quality_keyword_alert"],
                )
                alerts.append(alert)

        return alerts

    def detect_tiktok_spike(
        self,
        df: pd.DataFrame,
    ) -> list[CrisisAlert]:
        """
        Detect TikTok mention volume growing >200% week-over-week.
        """
        alerts = []
        tiktok = df[df["platform"] == "tiktok_comment"].copy()
        if tiktok.empty:
            return alerts

        tiktok["week"] = tiktok["timestamp"].dt.isocalendar().week.astype(int)
        tiktok["year"] = tiktok["timestamp"].dt.year
        weekly = tiktok.groupby(["year", "week"]).size().reset_index(name="count")
        weekly = weekly.sort_values(["year", "week"]).reset_index(drop=True)

        if len(weekly) < 2:
            return alerts

        for i in range(1, len(weekly)):
            prev = weekly.iloc[i - 1]["count"]
            curr = weekly.iloc[i]["count"]
            if prev > 0:
                growth = (curr - prev) / prev
                if growth > self.tiktok_wow_threshold:
                    alert = CrisisAlert(
                        alert_id=self._next_alert_id(),
                        timestamp=str(datetime.now()),
                        severity=SEVERITY_MAP["tiktok_growth_spike"],
                        alert_type="tiktok_growth_spike",
                        description=(
                            f"TikTok volume spike: Week {int(weekly.iloc[i]['week'])} "
                            f"has {curr} mentions vs {prev} previous week "
                            f"({growth * 100:.0f}% growth)."
                        ),
                        affected_platforms=["tiktok_comment"],
                        mention_count=curr,
                        suggested_action=SUGGESTED_ACTIONS["tiktok_growth_spike"],
                    )
                    alerts.append(alert)

        return alerts

    def detect_returns_spike(
        self,
        df: pd.DataFrame,
    ) -> list[CrisisAlert]:
        """
        Detect SKU return rate increasing >5 percentage points in 7 days.
        """
        alerts = []
        returns_df = df[df["platform"] == "returns_feedback_form"].copy()
        if returns_df.empty:
            return alerts

        returns_df["date"] = returns_df["timestamp"].dt.date

        for sku, group in returns_df.groupby("product_sku"):
            daily = group.groupby("date").size().reset_index(name="count")
            daily = daily.sort_values("date")

            if len(daily) < 14:
                continue

            # Compare last 7 days to previous 7 days
            recent = daily.tail(7)["count"].sum()
            previous = daily.iloc[-14:-7]["count"].sum()

            if previous > 0:
                diff_pp = ((recent - previous) / max(previous, 1)) * 100
                if diff_pp > self.returns_spike_threshold * 10:  # Rough proxy
                    alert = CrisisAlert(
                        alert_id=self._next_alert_id(),
                        timestamp=str(datetime.now()),
                        severity=SEVERITY_MAP["returns_spike"],
                        alert_type="returns_spike",
                        description=(
                            f"Returns spike for {sku}: {recent} returns in last 7 days "
                            f"vs {previous} in prior 7 days ({diff_pp:.0f}% increase)."
                        ),
                        affected_skus=[sku],
                        affected_platforms=["returns_feedback_form"],
                        mention_count=recent,
                        suggested_action=SUGGESTED_ACTIONS["returns_spike"],
                    )
                    alerts.append(alert)

        return alerts

    # ----- Main scan -----

    def scan(
        self,
        df: pd.DataFrame,
        sentiment_col: str = "sentiment",
    ) -> list[CrisisAlert]:
        """
        Run all crisis detection checks on the mentions DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: timestamp, platform, text, product_sku,
            author_follower_count, and a sentiment column.
        sentiment_col : str
            Name of the sentiment column.

        Returns
        -------
        list[CrisisAlert]
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        all_alerts: list[CrisisAlert] = []

        # 1. Negative volume spikes
        all_alerts.extend(self.detect_negative_volume_spike(df, sentiment_col))

        # 2. Viral risk
        all_alerts.extend(self.detect_viral_risk(df, sentiment_col))

        # 3. Quality keyword alerts
        all_alerts.extend(self.detect_quality_keyword_alerts(df))

        # 4. TikTok growth spikes
        all_alerts.extend(self.detect_tiktok_spike(df))

        # 5. Returns spikes
        all_alerts.extend(self.detect_returns_spike(df))

        # Sort by severity then timestamp
        all_alerts.sort(key=lambda a: (a.severity, a.timestamp))

        self.active_alerts = all_alerts

        # Persist to log
        self._log_alerts(all_alerts)

        # Simulate Slack
        for alert in all_alerts:
            self._simulate_slack_notification(alert)

        return all_alerts

    def _log_alerts(self, alerts: list[CrisisAlert]) -> None:
        """Persist alerts to JSON log file."""
        if not alerts:
            return

        log_file = self.crisis_log_dir / f"crisis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log_data = [asdict(a) for a in alerts]

        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2, default=str)

        logger.info(f"Logged {len(alerts)} alerts to {log_file}")

    def _simulate_slack_notification(self, alert: CrisisAlert) -> None:
        """Simulate sending a Slack notification."""
        icon = {1: "🚨", 2: "⚠️", 3: "📋"}.get(alert.severity, "ℹ️")
        msg = (
            f"{icon} [{alert.severity_label}] {alert.alert_type}\n"
            f"   {alert.description}\n"
            f"   → {alert.suggested_action}\n"
        )
        logger.warning(f"SLACK NOTIFICATION:\n{msg}")

    def get_active_alerts(self, max_severity: int = 3) -> list[CrisisAlert]:
        """Get active (unresolved) alerts up to a severity level."""
        return [a for a in self.active_alerts if a.severity <= max_severity and not a.resolved]

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                return True
        return False
