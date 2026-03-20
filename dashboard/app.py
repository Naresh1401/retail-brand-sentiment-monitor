"""
Streamlit dashboard for Retail Brand Sentiment Monitor.

Tabs:
1. Brand Health — NPS-style gauge, 30-day trend, volume by platform
2. Product Intelligence — Heatmap: SKU × topic × sentiment
3. Crisis Radar — Alert feed, viral risk monitor, severity badges
4. Returns Insights — Reason breakdown, SKU flags, quality issues
5. Competitive Share of Voice — Brand vs competitors
6. Regional Lens — Sentiment by region
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.returns_analysis import (
    analyze_returns,
    extract_quality_issues,
    return_reason_breakdown,
)
from src.crisis_detector import CrisisDetector, QUALITY_ALERT_KEYWORDS
from src.topics import RETAIL_SEED_TOPICS, keyword_topic_classify

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Retail Brand Sentiment Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
DATA_PATH = ROOT / "data" / "retail_mentions.csv"
COMPETITOR_BRANDS = ["RivalFit", "UrbanGlow", "PeakWear", "StyleVault", "FlexNova"]


@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error("Dataset not found. Run `python data/generate_data.py` first.")
        st.stop()

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    # Infer sentiment from star rating / keywords if not pre-labelled
    if "sentiment" not in df.columns:
        df["sentiment"] = df.apply(_infer_sentiment, axis=1)

    # Add topic labels
    if "topic" not in df.columns:
        df["topic"] = df["text"].apply(lambda t: keyword_topic_classify(t).topic_label)

    return df


def _infer_sentiment(row: pd.Series) -> str:
    star = row.get("star_rating")
    if pd.notna(star):
        if star <= 2:
            return "negative"
        elif star == 3:
            return "neutral"
        return "positive"
    text = str(row.get("text", "")).lower()
    neg = ["disappointed", "terrible", "awful", "worst", "poor", "defect", "broke", "waste",
           "garbage", "rubbish", "scam", "annoyed", "avoid", "falls apart", "stitching"]
    pos = ["love", "amazing", "perfect", "great", "excellent", "best", "happy",
           "obsessed", "incredible", "gorgeous", "stunning", "brilliant", "fantastic"]
    n = sum(1 for w in neg if w in text)
    p = sum(1 for w in pos if w in text)
    if n > p:
        return "negative"
    elif p > n:
        return "positive"
    return "neutral"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🛍️ Brand Monitor")
st.sidebar.markdown("---")

df = load_data()

date_range = st.sidebar.date_input(
    "Date range",
    value=(df["timestamp"].min().date(), df["timestamp"].max().date()),
    min_value=df["timestamp"].min().date(),
    max_value=df["timestamp"].max().date(),
)

if len(date_range) == 2:
    mask = (df["timestamp"].dt.date >= date_range[0]) & (df["timestamp"].dt.date <= date_range[1])
    df_filtered = df[mask].copy()
else:
    df_filtered = df.copy()

platforms = st.sidebar.multiselect(
    "Platforms",
    options=sorted(df["platform"].unique()),
    default=sorted(df["platform"].unique()),
)
df_filtered = df_filtered[df_filtered["platform"].isin(platforms)]

categories = st.sidebar.multiselect(
    "Product categories",
    options=sorted(df["product_category"].unique()),
    default=sorted(df["product_category"].unique()),
)
df_filtered = df_filtered[df_filtered["product_category"].isin(categories)]

st.sidebar.markdown(f"**{len(df_filtered):,}** mentions selected")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Brand Health",
    "🔍 Product Intelligence",
    "🚨 Crisis Radar",
    "📦 Returns Insights",
    "🏆 Competitive SOV",
    "🌍 Regional Lens",
])

# ========================== TAB 1: BRAND HEALTH ============================
with tab1:
    st.header("Brand Health Overview")

    col1, col2, col3, col4 = st.columns(4)
    total = len(df_filtered)
    pos = (df_filtered["sentiment"] == "positive").sum()
    neg = (df_filtered["sentiment"] == "negative").sum()
    neu = (df_filtered["sentiment"] == "neutral").sum()
    net = ((pos - neg) / total * 100) if total > 0 else 0

    col1.metric("Total Mentions", f"{total:,}")
    col2.metric("Positive", f"{pos:,}", f"{pos / total * 100:.1f}%" if total else "0%")
    col3.metric("Negative", f"{neg:,}", f"{neg / total * 100:.1f}%" if total else "0%")
    col4.metric("Net Sentiment", f"{net:+.1f}%")

    # NPS-style gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=net,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Net Sentiment Score", "font": {"size": 20}},
        gauge={
            "axis": {"range": [-100, 100], "tickwidth": 1},
            "bar": {"color": "#2563eb"},
            "steps": [
                {"range": [-100, -25], "color": "#fee2e2"},
                {"range": [-25, 25], "color": "#fef9c3"},
                {"range": [25, 100], "color": "#dcfce7"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 0,
            },
        },
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # 30-day rolling trend
    daily = df_filtered.set_index("timestamp").resample("D")["sentiment"].value_counts().unstack(fill_value=0)
    if not daily.empty:
        for col in ["positive", "neutral", "negative"]:
            if col not in daily.columns:
                daily[col] = 0
        daily["net"] = ((daily["positive"] - daily["negative"]) / daily.sum(axis=1) * 100)
        daily["net_rolling"] = daily["net"].rolling(7, min_periods=1).mean()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=daily.index, y=daily["net_rolling"],
            mode="lines", name="Net Sentiment (7-day MA)",
            line={"color": "#2563eb", "width": 3},
        ))
        fig_trend.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_trend.update_layout(
            title="Rolling 7-Day Net Sentiment Trend",
            yaxis_title="Net Sentiment %",
            height=350,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Volume by platform
    platform_vol = df_filtered["platform"].value_counts().reset_index()
    platform_vol.columns = ["platform", "count"]
    fig_vol = px.bar(
        platform_vol,
        x="platform",
        y="count",
        color="platform",
        title="Mention Volume by Platform",
    )
    fig_vol.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_vol, use_container_width=True)

# ======================== TAB 2: PRODUCT INTELLIGENCE ======================
with tab2:
    st.header("Product Intelligence")
    st.markdown("**Heatmap: Which products generate which types of complaints?**")

    # Build SKU × topic × sentiment matrix
    pivot_data = df_filtered.groupby(["product_sku", "topic", "sentiment"]).size().reset_index(name="count")

    # Focus on negative for issue detection
    neg_pivot = pivot_data[pivot_data["sentiment"] == "negative"]
    if not neg_pivot.empty:
        heatmap_df = neg_pivot.pivot_table(index="product_sku", columns="topic", values="count", fill_value=0)

        # Top 20 SKUs by total negative mentions
        top_skus = neg_pivot.groupby("product_sku")["count"].sum().nlargest(20).index
        heatmap_df = heatmap_df.loc[heatmap_df.index.isin(top_skus)]

        if not heatmap_df.empty:
            fig_heat = px.imshow(
                heatmap_df,
                labels={"x": "Topic", "y": "SKU", "color": "Negative Mentions"},
                title="Negative Mentions: SKU × Topic Heatmap (Top 20 SKUs)",
                color_continuous_scale="Reds",
                aspect="auto",
            )
            fig_heat.update_layout(height=600)
            st.plotly_chart(fig_heat, use_container_width=True)

    # Topic distribution
    topic_dist = df_filtered["topic"].value_counts().reset_index()
    topic_dist.columns = ["topic", "count"]
    fig_topic = px.pie(
        topic_dist, values="count", names="topic",
        title="Topic Distribution Across All Mentions",
    )
    st.plotly_chart(fig_topic, use_container_width=True)

    # Sentiment by topic
    topic_sent = df_filtered.groupby(["topic", "sentiment"]).size().reset_index(name="count")
    fig_ts = px.bar(
        topic_sent, x="topic", y="count", color="sentiment",
        title="Sentiment Breakdown by Topic",
        color_discrete_map={"positive": "#22c55e", "neutral": "#eab308", "negative": "#ef4444"},
        barmode="group",
    )
    st.plotly_chart(fig_ts, use_container_width=True)

# ========================= TAB 3: CRISIS RADAR ============================
with tab3:
    st.header("🚨 Crisis Radar")

    detector = CrisisDetector(crisis_log_dir=str(ROOT / "crisis_logs"))
    alerts = detector.scan(df_filtered, sentiment_col="sentiment")

    if not alerts:
        st.success("✅ No active crisis alerts. All clear.")
    else:
        # Summary metrics
        sev1 = sum(1 for a in alerts if a.severity == 1)
        sev2 = sum(1 for a in alerts if a.severity == 2)
        sev3 = sum(1 for a in alerts if a.severity == 3)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Alerts", len(alerts))
        c2.metric("🔴 Sev-1 Critical", sev1)
        c3.metric("🟠 Sev-2 High", sev2)
        c4.metric("🟡 Sev-3 Medium", sev3)

        st.markdown("---")

        # Alert feed
        for alert in alerts[:30]:
            severity_colors = {1: "🔴", 2: "🟠", 3: "🟡"}
            icon = severity_colors.get(alert.severity, "⚪")

            with st.expander(
                f"{icon} [{alert.severity_label}] {alert.alert_type} — {alert.timestamp[:19]}",
                expanded=alert.severity == 1,
            ):
                st.markdown(f"**Description:** {alert.description}")
                if alert.affected_skus:
                    st.markdown(f"**Affected SKUs:** {', '.join(alert.affected_skus)}")
                if alert.affected_platforms:
                    st.markdown(f"**Platforms:** {', '.join(alert.affected_platforms)}")
                st.markdown(f"**Mentions:** {alert.mention_count}")
                st.info(f"**Suggested Action:** {alert.suggested_action}")

    # Viral risk monitor
    st.subheader("Viral Risk Monitor")
    social = df_filtered[
        df_filtered["platform"].isin(["tiktok_comment", "instagram_comment", "twitter_mention"])
    ].copy()
    if not social.empty:
        high_reach = social[social["author_follower_count"].fillna(0) > 10_000].copy()
        if not high_reach.empty:
            high_reach_neg = high_reach[high_reach["sentiment"] == "negative"]
            st.metric(
                "High-Reach Negative Posts (>10K followers)",
                f"{len(high_reach_neg)}",
            )
            if not high_reach_neg.empty:
                st.dataframe(
                    high_reach_neg[["timestamp", "platform", "text", "product_sku", "author_follower_count"]]
                    .sort_values("author_follower_count", ascending=False)
                    .head(10),
                    use_container_width=True,
                )
        else:
            st.info("No high-reach posts detected.")

# ======================== TAB 4: RETURNS INSIGHTS ==========================
with tab4:
    st.header("📦 Returns Insights")

    returns_result = analyze_returns(df_filtered)

    # Return reason breakdown
    st.subheader("Return Reason Breakdown")
    if not returns_result.return_reason_breakdown.empty:
        fig_rr = px.bar(
            returns_result.return_reason_breakdown,
            x="product_category",
            y="count",
            color="return_reason",
            title="Return Reasons by Product Category",
            barmode="stack",
        )
        st.plotly_chart(fig_rr, use_container_width=True)

    # SKU flagging
    st.subheader("SKU Alerts")
    if returns_result.alerts:
        for alert in returns_result.alerts:
            color = "🔴" if alert.alert_type == "high_return_rate" else "🟡"
            st.warning(f"{color} **{alert.sku}** — {alert.alert_type}: {alert.recommendation}")
    else:
        st.success("No SKU alerts triggered.")

    # Size issue SKUs
    if returns_result.size_issue_skus:
        st.subheader("Size Issue Hotspots")
        for sku in returns_result.size_issue_skus:
            st.error(f"📏 **{sku}** — Update size guide recommended (>30% size-issue returns)")

    # Quality issue extraction
    st.subheader("NLP Quality Issue Extraction")
    if returns_result.quality_issues:
        qi_df = pd.DataFrame([
            {"issue_type": qi.issue_type, "matched_text": qi.matched_text}
            for qi in returns_result.quality_issues
        ])
        issue_counts = qi_df["issue_type"].value_counts().reset_index()
        issue_counts.columns = ["issue_type", "count"]

        fig_qi = px.bar(issue_counts, x="issue_type", y="count", title="Quality Issues Extracted from Returns Text")
        st.plotly_chart(fig_qi, use_container_width=True)

        st.dataframe(qi_df.head(20), use_container_width=True)

    # Recommendations
    st.subheader("Recommendations")
    for rec in returns_result.recommendations:
        st.markdown(f"- {rec}")

# ====================== TAB 5: COMPETITIVE SOV ============================
with tab5:
    st.header("🏆 Competitive Share of Voice")
    st.markdown("*Simulated competitor data for comparison*")

    # Generate synthetic competitor data
    np.random.seed(42)
    our_vol = len(df_filtered)
    our_net = ((df_filtered["sentiment"] == "positive").sum() - (df_filtered["sentiment"] == "negative").sum()) / max(our_vol, 1) * 100

    comp_data = [{"brand": "Our Brand", "mention_volume": our_vol, "net_sentiment": round(our_net, 1)}]
    for comp in COMPETITOR_BRANDS:
        vol = int(our_vol * np.random.uniform(0.3, 1.2))
        net = round(np.random.uniform(-15, 45), 1)
        comp_data.append({"brand": comp, "mention_volume": vol, "net_sentiment": net})

    comp_df = pd.DataFrame(comp_data)

    col1, col2 = st.columns(2)
    with col1:
        fig_sov = px.pie(comp_df, values="mention_volume", names="brand", title="Share of Voice (Mention Volume)")
        st.plotly_chart(fig_sov, use_container_width=True)

    with col2:
        fig_sent = px.bar(
            comp_df.sort_values("net_sentiment"),
            x="net_sentiment",
            y="brand",
            orientation="h",
            title="Net Sentiment Comparison",
            color="net_sentiment",
            color_continuous_scale="RdYlGn",
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    st.dataframe(comp_df.sort_values("mention_volume", ascending=False), use_container_width=True)

# ======================== TAB 6: REGIONAL LENS =============================
with tab6:
    st.header("🌍 Regional Sentiment Lens")

    region_sent = df_filtered.groupby(["region", "sentiment"]).size().reset_index(name="count")
    region_totals = df_filtered.groupby("region").size().reset_index(name="total")

    region_net = df_filtered.groupby("region")["sentiment"].apply(
        lambda x: ((x == "positive").sum() - (x == "negative").sum()) / len(x) * 100
    ).reset_index(name="net_sentiment")

    # Net sentiment by region
    fig_region = px.bar(
        region_net.sort_values("net_sentiment"),
        x="net_sentiment",
        y="region",
        orientation="h",
        title="Net Sentiment by Region",
        color="net_sentiment",
        color_continuous_scale="RdYlGn",
    )
    fig_region.update_layout(height=300)
    st.plotly_chart(fig_region, use_container_width=True)

    # Detailed breakdown
    fig_region_detail = px.bar(
        region_sent,
        x="region",
        y="count",
        color="sentiment",
        title="Sentiment Distribution by Region",
        color_discrete_map={"positive": "#22c55e", "neutral": "#eab308", "negative": "#ef4444"},
        barmode="group",
    )
    st.plotly_chart(fig_region_detail, use_container_width=True)

    # Regional topic comparison
    st.subheader("Topic Hotspots by Region")
    region_topic = df_filtered[df_filtered["sentiment"] == "negative"].groupby(
        ["region", "topic"]
    ).size().reset_index(name="count")

    if not region_topic.empty:
        fig_rt = px.bar(
            region_topic,
            x="topic",
            y="count",
            color="region",
            title="Negative Mention Topics by Region",
            barmode="group",
        )
        st.plotly_chart(fig_rt, use_container_width=True)

    # Question: Are US customers more negative about sizing?
    st.subheader("Sizing Complaints: Regional Comparison")
    sizing = df_filtered[df_filtered["topic"] == "sizing_and_fit"]
    if not sizing.empty:
        sizing_region = sizing.groupby(["region", "sentiment"]).size().reset_index(name="count")
        fig_sizing = px.bar(
            sizing_region,
            x="region",
            y="count",
            color="sentiment",
            title="Sizing & Fit Mentions by Region",
            color_discrete_map={"positive": "#22c55e", "neutral": "#eab308", "negative": "#ef4444"},
            barmode="stack",
        )
        st.plotly_chart(fig_sizing, use_container_width=True)
