"""
Product-level issue extraction using BERTopic with retail-specific seed topics.

Extracts topic clusters from brand mentions, maps them to actionable retail
categories, and returns per-mention topic labels with sentiment overlay.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------------------------------------------------------
# Retail seed topics — keywords that anchor BERTopic clusters
# ---------------------------------------------------------------------------
RETAIL_SEED_TOPICS = {
    "sizing_and_fit": [
        "runs small", "too big", "sizing off", "size guide", "tight",
        "loose", "fit", "true to size", "oversized", "narrow", "wide",
        "length", "too short", "too long", "size chart",
    ],
    "quality": [
        "stitching", "material", "durability", "pilling", "cheap",
        "thin", "see-through", "quality", "fabric", "threads",
        "falling apart", "broke", "ripped", "defect", "poor quality",
    ],
    "delivery": [
        "late", "packaging", "damaged in transit", "shipping", "arrived",
        "delayed", "tracking", "lost parcel", "delivery time", "courier",
        "dispatch", "postage",
    ],
    "customer_service": [
        "returns process", "refund", "response time", "customer service",
        "support", "email", "chat", "unhelpful", "waited", "no reply",
        "exchange", "complaint",
    ],
    "style_and_design": [
        "colour", "color", "design", "looks different", "photo",
        "pattern", "style", "aesthetic", "trendy", "looks cheap",
        "beautiful", "gorgeous", "ugly",
    ],
    "price_and_value": [
        "overpriced", "worth it", "good value", "expensive", "cheap",
        "bargain", "price", "cost", "money", "affordable", "premium",
        "rip off", "value for money",
    ],
}

# Flatten for guided topic modelling
SEED_TOPIC_LIST: list[list[str]] = list(RETAIL_SEED_TOPICS.values())
TOPIC_LABELS: list[str] = list(RETAIL_SEED_TOPICS.keys())


@dataclass
class TopicResult:
    """Topic assignment for a single mention."""
    topic_id: int
    topic_label: str
    keywords: list[str]
    confidence: float


@dataclass
class TopicModelOutput:
    """Full output from topic modelling on a corpus."""
    mention_topics: list[TopicResult]
    topic_summary: pd.DataFrame  # topic_id, label, count, top_keywords
    model: object = field(repr=False, default=None)


class RetailTopicExtractor:
    """
    BERTopic-based product issue extractor with retail seed topics.

    Flow:
    1. Embed mentions using sentence-transformers
    2. Cluster via HDBSCAN with guided seed topics
    3. Map clusters → retail issue categories
    4. Return per-mention topic + keywords
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        min_topic_size: int = 30,
        n_gram_range: tuple[int, int] = (1, 3),
        seed_topic_list: list[list[str]] | None = None,
    ):
        self.embedding_model = embedding_model
        self.min_topic_size = min_topic_size
        self.n_gram_range = n_gram_range
        self.seed_topic_list = seed_topic_list or SEED_TOPIC_LIST
        self._model = None

    def _build_model(self):
        """Construct the BERTopic model with retail settings."""
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from umap import UMAP

        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_topic_size,
            min_samples=10,
            metric="euclidean",
            prediction_data=True,
        )
        vectorizer = CountVectorizer(
            ngram_range=self.n_gram_range,
            stop_words="english",
            max_features=10_000,
        )

        self._model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            seed_topic_list=self.seed_topic_list,
            nr_topics="auto",
            verbose=False,
        )
        return self._model

    @property
    def model(self):
        if self._model is None:
            self._build_model()
        return self._model

    def _map_topic_to_retail_label(self, topic_keywords: list[str]) -> str:
        """
        Map a BERTopic cluster to the closest retail seed topic by keyword overlap.
        """
        best_label = "other"
        best_score = 0

        kw_set = set(w.lower() for w in topic_keywords)

        for label, seeds in RETAIL_SEED_TOPICS.items():
            seed_tokens = set()
            for phrase in seeds:
                for token in phrase.lower().split():
                    seed_tokens.add(token)

            overlap = len(kw_set & seed_tokens)
            if overlap > best_score:
                best_score = overlap
                best_label = label

        return best_label if best_score > 0 else "other"

    def extract(
        self,
        texts: list[str],
        embeddings: np.ndarray | None = None,
    ) -> TopicModelOutput:
        """
        Run topic extraction on a list of texts.

        Parameters
        ----------
        texts : list[str]
            Mention texts.
        embeddings : np.ndarray or None
            Pre-computed embeddings (optional).

        Returns
        -------
        TopicModelOutput
        """
        topics, probs = self.model.fit_transform(texts, embeddings=embeddings)

        topic_info = self.model.get_topic_info()

        # Build per-topic metadata
        topic_label_map = {}
        topic_keyword_map = {}

        for _, row in topic_info.iterrows():
            tid = row["Topic"]
            if tid == -1:
                topic_label_map[tid] = "outlier"
                topic_keyword_map[tid] = []
                continue

            topic_words = self.model.get_topic(tid)
            keywords = [w for w, _ in topic_words[:10]]
            label = self._map_topic_to_retail_label(keywords)
            topic_label_map[tid] = label
            topic_keyword_map[tid] = keywords

        # Per-mention results
        mention_topics = []
        for i, tid in enumerate(topics):
            conf = float(probs[i]) if probs is not None and len(probs) > i else 0.0
            # Handle array probs (soft clustering)
            if hasattr(conf, "__len__"):
                conf = float(max(probs[i])) if len(probs[i]) > 0 else 0.0

            mention_topics.append(
                TopicResult(
                    topic_id=int(tid),
                    topic_label=topic_label_map.get(tid, "other"),
                    keywords=topic_keyword_map.get(tid, [])[:5],
                    confidence=round(conf, 4),
                )
            )

        # Summary table
        summary_rows = []
        for tid in sorted(topic_label_map.keys()):
            if tid == -1:
                continue
            count = topics.count(tid) if isinstance(topics, list) else int((np.array(topics) == tid).sum())
            summary_rows.append(
                {
                    "topic_id": tid,
                    "label": topic_label_map[tid],
                    "count": count,
                    "top_keywords": ", ".join(topic_keyword_map[tid][:5]),
                }
            )

        summary_df = pd.DataFrame(summary_rows)

        return TopicModelOutput(
            mention_topics=mention_topics,
            topic_summary=summary_df,
            model=self.model,
        )


# ---------------------------------------------------------------------------
# Lightweight keyword-based fallback (no GPU required, good for API)
# ---------------------------------------------------------------------------

def keyword_topic_classify(text: str) -> TopicResult:
    """
    Fast rule-based topic classifier using seed keywords.
    Useful for single-request API calls where BERTopic overhead is too high.
    """
    text_lower = text.lower()
    scores: dict[str, int] = {}

    for label, keywords in RETAIL_SEED_TOPICS.items():
        score = 0
        matched_kw = []
        for kw in keywords:
            if kw.lower() in text_lower:
                score += 1
                matched_kw.append(kw)
        if score > 0:
            scores[label] = score

    if not scores:
        return TopicResult(
            topic_id=-1,
            topic_label="other",
            keywords=[],
            confidence=0.0,
        )

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]
    total_kw = len(RETAIL_SEED_TOPICS[best_label])
    confidence = min(best_score / max(total_kw * 0.3, 1), 1.0)

    matched = [kw for kw in RETAIL_SEED_TOPICS[best_label] if kw.lower() in text_lower]

    return TopicResult(
        topic_id=TOPIC_LABELS.index(best_label),
        topic_label=best_label,
        keywords=matched[:5],
        confidence=round(confidence, 4),
    )
