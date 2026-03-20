# 🛍️ Retail Brand Sentiment Monitor

**How a TikTok quality complaint becomes a £2M returns problem in 72 hours**

> A single TikTok video about a stitching defect reaches 500K views → gets picked up
> on Twitter and Reddit → 1-star Trustpilot reviews flood in → returns spike 300% →
> product team scrambles to pull inventory → brand reputation takes months to recover.

This system catches that cascade at Stage 1.

---

## The 4 Levels of Retail Brand Risk

| Level | Signal | Window | Example |
|-------|--------|--------|---------|
| **1. Mention** | Single negative post | Real-time | "This hoodie stitching is awful" |
| **2. Trend** | Volume uptick on topic | Hours | 50+ complaints about same SKU |
| **3. Crisis** | Cross-platform viral spike | 4-24 hours | TikTok video + Twitter pile-on |
| **4. Recall** | Safety/quality threshold | Days | "Defect", "injury", "toxic" keywords |

Our system monitors all four levels continuously.

---

## What This System Does

- **Sentiment classification** on reviews, social mentions, and returns feedback using `cardiffnlp/twitter-roberta-base-sentiment-latest` (fine-tuned thresholds for retail domain)
- **Product-level issue extraction** via BERTopic with retail-specific seed topics (sizing, quality, delivery, service, style, price)
- **Returns analysis** with automated SKU flagging — size issues, quality defects, return rate alerts
- **Crisis detection** across 5 signal types: volume spikes, viral risk, quality keywords, TikTok growth, returns spikes
- **Real-time dashboard** with 6 views: brand health, product intelligence, crisis radar, returns insights, competitive SOV, regional lens

---

## Project Structure

```
retail-brand-sentiment-monitor/
├── data/
│   ├── generate_data.py          # Synthetic 12K mentions generator
│   └── retail_mentions.csv       # Generated dataset (gitignored)
├── src/
│   ├── sentiment.py              # Retail-tuned sentiment classifier
│   ├── topics.py                 # BERTopic product issue extraction
│   ├── returns_analysis.py       # Returns analysis & SKU flagging
│   └── crisis_detector.py        # 5-signal crisis detection engine
├── api/
│   └── main.py                   # FastAPI service (5 endpoints)
├── dashboard/
│   └── app.py                    # Streamlit dashboard (6 tabs)
├── notebooks/
│   └── model_eval.ipynb          # 300-sample evaluation notebook
├── tests/
│   ├── test_sentiment.py
│   ├── test_topics.py
│   ├── test_returns.py
│   ├── test_crisis.py
│   └── test_api.py
├── crisis_logs/                  # Auto-generated alert logs (gitignored)
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset

```bash
python data/generate_data.py
```

This creates 12,800 mentions (12,000 normal + 800 crisis injection in month 4) with:
- 8 platforms (Trustpilot, Google Reviews, Instagram, TikTok, Twitter, Reddit, Returns, Surveys)
- 5 product categories (apparel, footwear, accessories, beauty, homeware)
- Crisis event: SKU `APP-0017` (Flex-Fit Performance Hoodie) — stitching defect going viral

### 3. Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

Endpoints:
- `POST /analyze` — Sentiment + topic + crisis flag for any text
- `POST /analyze-return` — Return reason classification + quality issue extraction
- `GET /brand-health` — Rolling 7-day net sentiment by channel
- `GET /sku-alerts` — Flagged SKUs (high return rate / negative sentiment)
- `GET /crisis-status` — Active crisis alerts with severity

### 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## Model Card

### cardiffnlp/twitter-roberta-base-sentiment-latest

| Property | Detail |
|----------|--------|
| **Base model** | RoBERTa (125M params) |
| **Training data** | ~124M tweets, fine-tuned on TweetEval sentiment |
| **Output classes** | positive, neutral, negative |
| **Retail adaptations** | Emoji pre-processing, star-rating calibration, threshold tuning |

### Performance (on 300 gold-standard retail reviews)

| Metric | Positive | Neutral | Negative |
|--------|----------|---------|----------|
| Precision | ~0.90 | ~0.72 | ~0.88 |
| Recall | ~0.92 | ~0.68 | ~0.85 |
| F1 | ~0.91 | ~0.70 | ~0.86 |

**Overall accuracy: ~82-85%** (varies with sarcasm prevalence)

### Known Limitations

1. **Sarcasm**: "Oh great, another parcel that arrived in pieces 🙃" → often misclassified as positive
2. **UK slang**: Informal British English may confuse the model
3. **Short text**: <5 word inputs have lower confidence
4. **Domain gap**: Model trained on tweets, not product reviews — calibration helps but doesn't eliminate gap

---

## Key Finding

> **Trustpilot 3-star reviews are the most information-dense.** They contain specific,
> actionable product feedback that 1-star emotional rants lack. Our topic extraction
> module targets these for maximum product insight per mention.

---

## Crisis Detection Signals

| Signal | Trigger | Severity | Action |
|--------|---------|----------|--------|
| Quality keyword | "defect", "toxic", "injury", "recall" | Sev-1 | Escalate to product safety |
| Vol spike | >50 negative mentions in 4h | Sev-2 | Brief CS team, prepare FAQ |
| Viral risk | >10K follower + negative + product tag | Sev-2 | Social team rapid response |
| TikTok spike | >200% WoW growth | Sev-2 | Monitor & prepare content |
| Returns spike | >5pp increase in 7 days | Sev-3 | Pull QC reports |

---

## Top 3 Flagged SKUs (Synthetic Data)

| SKU | Issue | Recommended Action |
|-----|-------|--------------------|
| **APP-0017** | Stitching defect crisis (800 mentions in 2 weeks) | Pull from sale. Issue recall. Prepare public statement. |
| **FTW-0012** | High return rate (sizing inconsistency) | Update size guide. Add "runs small" warning. |
| **BTY-0023** | Quality complaints (skin reactions) | Review ingredients. Add allergen warnings. |

---

## Connecting to Real Data Sources

### Trustpilot API
```python
# Trustpilot Business API (requires business account)
# GET https://api.trustpilot.com/v1/business-units/{businessUnitId}/reviews
# Headers: Authorization: Bearer {api_key}
```

### Instagram Graph API
```python
# Instagram Graph API (requires Facebook Business)
# GET https://graph.facebook.com/v18.0/{ig-user-id}/media
# Params: fields=id,caption,timestamp,comments_count
```

### Reddit API (PRAW)
```python
import praw
reddit = praw.Reddit(client_id="...", client_secret="...", user_agent="...")
subreddit = reddit.subreddit("fashionreps+streetwear")
for submission in subreddit.search("brand_name", limit=100):
    process(submission.title, submission.selftext)
```

### Twitter/X API v2
```python
# GET https://api.twitter.com/2/tweets/search/recent
# Params: query=brand_name -is:retweet&tweet.fields=created_at,public_metrics
```

---

## Returns Intelligence

The returns analysis module (`src/returns_analysis.py`) provides:

1. **Return rate monitoring**: Auto-flag SKUs with >15% return rate
2. **Size issue detection**: Flag SKUs where >30% returns cite sizing → generates "Update size guide for SKU-XXXX" recommendation
3. **NLP quality extraction**: Regex-based extraction of 7 defect categories from free-text returns feedback:
   - Stitching defects
   - Material quality
   - Pilling
   - Colour fade
   - Zip/button defects
   - Sizing inconsistency
   - Damage on arrival

---

## Tech Stack

- **Python 3.11**
- **transformers** (HuggingFace) — sentiment classification
- **sentence-transformers** — text embeddings for BERTopic
- **BERTopic** — topic modelling with guided seed topics
- **pandas / numpy** — data manipulation
- **FastAPI** — NLP API service
- **Streamlit** — interactive dashboard
- **plotly** — visualisations
- **pytest** — testing

---

## License

MIT

---

*Built for DTC retail brands who refuse to let a stitching defect become a PR crisis.*
