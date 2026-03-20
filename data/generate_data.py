"""
Synthetic retail mentions data generator.

Generates 12,000 realistic brand mentions across 8 platforms over 6 months,
including a product quality crisis injection in month 4.
"""

from __future__ import annotations

import random
import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PLATFORMS = [
    "trustpilot_review",
    "google_review",
    "instagram_comment",
    "tiktok_comment",
    "twitter_mention",
    "reddit_post",
    "returns_feedback_form",
    "post_purchase_survey",
]

PLATFORM_WEIGHTS = [0.15, 0.10, 0.18, 0.15, 0.12, 0.08, 0.12, 0.10]

PRODUCT_CATEGORIES = ["apparel", "footwear", "accessories", "beauty", "homeware"]
CATEGORY_WEIGHTS = [0.30, 0.20, 0.15, 0.20, 0.15]

REGIONS = ["UK", "US", "EU", "APAC"]
REGION_WEIGHTS = [0.30, 0.35, 0.20, 0.15]

ORDER_VALUE_BANDS = ["low", "mid", "high"]
VALUE_BAND_WEIGHTS = [0.30, 0.45, 0.25]

RETURN_REASONS = [
    "size_issue",
    "quality",
    "not_as_described",
    "changed_mind",
    "damaged",
]
RETURN_REASON_WEIGHTS = [0.30, 0.25, 0.15, 0.20, 0.10]

# SKU pools per category
SKUS = {
    "apparel": [f"APP-{i:04d}" for i in range(1, 51)],
    "footwear": [f"FTW-{i:04d}" for i in range(1, 31)],
    "accessories": [f"ACC-{i:04d}" for i in range(1, 26)],
    "beauty": [f"BTY-{i:04d}" for i in range(1, 36)],
    "homeware": [f"HMW-{i:04d}" for i in range(1, 21)],
}

# The crisis SKU
CRISIS_SKU = "APP-0017"
CRISIS_PRODUCT = "Flex-Fit Performance Hoodie"

# ---------------------------------------------------------------------------
# Realistic review / comment templates
# ---------------------------------------------------------------------------

POSITIVE_TEMPLATES = [
    "Absolutely love my new {product}! The quality is incredible for the price 🔥",
    "Best purchase I've made this year. The {product} fits perfectly ❤️",
    "Obsessed with this 🙏 the {product} is even better in person",
    "Just got my {product} and I'm so impressed. Will definitely be ordering more 💯",
    "Ship was super fast and the {product} quality is top notch. 10/10 would recommend",
    "The {product} exceeded my expectations. Great colour and really comfortable",
    "Five stars! My {product} arrived quickly and the material feels premium",
    "Can't stop wearing my new {product}! So many compliments already 😍",
    "Really good value for money. The {product} looks exactly like the photos",
    "Ordered the {product} for my partner and they absolutely loved it ❤️",
    "This {product} is honestly my favourite thing in my wardrobe right now",
    "So happy with this purchase! The {product} is beautifully made",
    "The packaging was lovely and the {product} itself is gorgeous 🔥",
    "Third time ordering from here and never disappointed. {product} is amazing",
    "Wearing my {product} right now and it's so comfy. Perfect for everyday",
    "Just received my order — the {product} is stunning quality 💯",
    "Such a great brand. My {product} arrived in perfect condition",
    "The {product} is worth every penny. Brilliant quality and design",
    "OMG the {product} is incredible!! Already planning my next order 🙌",
    "Finally found a brand that delivers quality. The {product} is chef's kiss 👨‍🍳",
]

NEUTRAL_TEMPLATES = [
    "The {product} is okay. Nothing special but does the job",
    "Decent {product} for the price. Not amazing but not bad either",
    "My {product} arrived on time. It's fine, pretty average quality",
    "The {product} is alright. Sizing is a bit off but otherwise okay",
    "Got the {product}. It's what I expected — nothing more nothing less",
    "The {product} is adequate. Might order again if there's a sale",
    "Mixed feelings about the {product}. The colour is nice but material is thin",
    "The {product} is standard quality. You get what you pay for I suppose",
    "It's a 3-star {product} for me. Some things good some things could be better",
    "Average {product}. The delivery was quick though so that's a plus",
    "The {product} looks slightly different from the website photos. Still okay though",
    "Not blown away by the {product} but it's serviceable. Might keep it",
    "The {product} is fine for casual wear. Wouldn't recommend for anything serious",
    "Ordered the {product} on a whim. It's meh to be honest",
    "The {product} quality is middle of the road. Expected a bit more",
]

NEGATIVE_TEMPLATES = [
    "Really disappointed with my {product}. The quality is terrible for the price 😭",
    "The {product} fell apart after one wash. Complete waste of money 💔",
    "Avoid the {product}. Sizing is completely off and the material feels cheap",
    "Waited 3 weeks for my {product} and it arrived damaged. Awful experience 🙅",
    "The {product} looks nothing like the photos online. Very misleading",
    "Terrible quality on the {product}. Stitching came undone within a week",
    "So annoyed — the {product} doesn't fit at all even though I checked the size guide",
    "Don't waste your money on the {product}. Overpriced for such poor quality",
    "The {product} is already pilling after two wears. Not impressed at all",
    "Customer service was useless when I tried to return my {product}. Never again",
    "The colour of my {product} faded after first wash. Rubbish quality 😤",
    "My {product} had a hole in it when it arrived. Quality control is non-existent",
    "Honestly shocked at how bad the {product} is. Going back to other brands",
    "The {product} is way too small. Their sizing chart is completely wrong",
    "Paid premium price for the {product} and got budget quality. Scam",
    "The {product} material is scratchy and uncomfortable. Very disappointing",
    "Ordered my {product} weeks ago and still waiting. No response from support either",
    "One star. The {product} is garbage. Thin material, bad stitching, wrong colour",
    "The zip on my {product} broke on day one. Absolute joke of a product",
    "Sent back the {product} immediately. Not even close to what was advertised",
]

# Crisis-specific templates (stitching defect / falls apart)
CRISIS_TEMPLATES = [
    "The stitching on my {product} is already coming apart after ONE wear. Absolute garbage 😭",
    "WARNING: The {product} has a major stitching defect. Falls apart within days 💔",
    "My {product} literally fell apart at the seams after the first wash. Stitching is defective",
    "Has anyone else had stitching issues with the {product}? Mine is falling apart 🙅",
    "DO NOT BUY the {product}. The stitching defect is real — mine lasted 3 days",
    "The {product} seams are coming undone everywhere. This is a quality control nightmare",
    "Just had my {product} fall apart mid-workout. The stitching is dangerously bad",
    "Stitching defect confirmed on my {product}. Third friend with the same issue 😤",
    "The {product} falls apart if you look at it wrong. Worst purchase ever. Stitching is a joke",
    "AVOID the {product}!!! Stitching defect — falls apart after one use. DEMAND A RECALL",
    "My {product} stitching ripped during a meeting. So embarrassing. Defective product",
    "The {product} literally disintegrated. Stitching quality is atrocious. Want my money back",
    "Second {product} with the same stitching defect. This is a systematic quality issue",
    "The {product} stitching came undone at the gym. Others in my class had the same issue",
    "Bought 2x {product} and BOTH have stitching defects. Falls apart = not fit for sale",
    "This {product} is dangerous — stitching defect means it falls apart during exercise",
    "Can't believe the {product} quality. Stitching defect makes it unwearable after a day",
    "POV: your brand new {product} falls apart because of a stitching defect 🤡",
    "The {product} stitching defect is going viral for a reason — AVOID",
    "Stitching on the {product} is appalling. Falls apart immediately. Quality crisis much?",
]

RETURN_FEEDBACK_TEMPLATES = {
    "size_issue": [
        "Returning the {product} — runs way too small. Ordered my usual size and it's tiny",
        "Size issue with {product}. The size guide doesn't match reality at all",
        "The {product} is massive even though I got my normal size. Sizing is off",
        "Doesn't fit. The {product} shoulders are too narrow. Size chart is wrong",
        "Returning because the {product} fits like it's two sizes smaller than labeled",
    ],
    "quality": [
        "Quality of the {product} is poor. Material feels cheap and thin",
        "The {product} has loose threads everywhere. Returning due to quality issues",
        "Stitching on the {product} is coming apart already. Very poor quality",
        "The {product} material started pilling immediately. Not acceptable quality",
        "Returning the {product} — the fabric is see-through. Quality is rubbish",
    ],
    "not_as_described": [
        "The {product} colour is completely different from the website photos",
        "Not as described — the {product} material is nothing like what was shown",
        "The {product} design doesn't match the listing at all. Misleading photos",
        "Returning the {product} because the texture is totally different to what's advertised",
        "The {product} looks nothing like the pictures. False advertising honestly",
    ],
    "changed_mind": [
        "Changed my mind about the {product}. Just not what I was looking for",
        "Decided the {product} isn't really my style after all. Returning it",
        "The {product} doesn't go with anything I own. Returning unused",
        "Found a better option elsewhere. Returning the {product} unopened",
        "Not sure why I ordered the {product}. Changed my mind, sending back",
    ],
    "damaged": [
        "The {product} arrived with a rip in the packaging and product damaged",
        "My {product} was crushed in transit. Box was completely flattened",
        "The {product} arrived with a stain on it. Must have been damaged in warehouse",
        "Returning the {product} — arrived broken. Delivery must have been rough",
        "The {product} packaging was torn and the item damaged. Disappointing",
    ],
}

SURVEY_TEMPLATES = {
    "positive": [
        "Great experience overall. The {product} was exactly what I wanted. 😊",
        "Really happy with my purchase. Fast delivery and the {product} is perfect",
        "Would recommend. The {product} is brilliant and the whole process was smooth",
        "Very satisfied with the {product}. Will shop here again for sure ❤️",
    ],
    "neutral": [
        "The {product} is okay. Delivery was fine. Average experience overall",
        "Nothing bad to say about the {product} but nothing amazing either",
        "Standard online shopping experience. {product} meets basic expectations",
    ],
    "negative": [
        "Not happy with the {product}. Expected better quality for the price paid",
        "Frustrated with the whole experience. {product} quality is subpar",
        "Won't be ordering again. The {product} was a letdown and delivery was slow",
    ],
}

PRODUCT_NAMES = {
    "apparel": [
        "Performance Hoodie", "Slim Fit Tee", "Cropped Sweatshirt",
        "High-Waist Leggings", "Oversized Jacket", "Ribbed Tank Top",
        "Utility Joggers", "Mesh Panel Top", "Longline Coat",
        "Seamless Sports Bra",
    ],
    "footwear": [
        "Training Shoe", "Running Sneaker", "Chunky Platform Boot",
        "Slip-On Loafer", "Knit Trainer", "Trail Runner",
        "Canvas Plimsoll", "Leather Chelsea Boot",
    ],
    "accessories": [
        "Canvas Tote Bag", "Beanie Hat", "Crossbody Bag",
        "Bucket Hat", "Chain Necklace", "Silk Scarf",
        "Sports Watch", "Leather Belt",
    ],
    "beauty": [
        "Hydrating Serum", "Matte Lipstick", "Foundation SPF30",
        "Vitamin C Moisturiser", "Setting Spray", "Brow Gel",
        "Cleansing Balm", "Retinol Night Cream",
    ],
    "homeware": [
        "Scented Candle", "Linen Cushion Cover", "Ceramic Vase",
        "Cotton Throw Blanket", "Reed Diffuser", "Trinket Tray",
    ],
}

COMPETITOR_BRANDS = ["RivalFit", "UrbanGlow", "PeakWear", "StyleVault", "FlexNova"]


def _generate_mention_id(index: int) -> str:
    """Deterministic mention ID from index."""
    return f"MNT-{hashlib.md5(str(index).encode()).hexdigest()[:10].upper()}"


def _pick_product_name(category: str) -> str:
    return random.choice(PRODUCT_NAMES[category])


def _generate_follower_count(platform: str) -> int | None:
    """Follower counts for social platforms; None for review/feedback."""
    social = {"instagram_comment", "tiktok_comment", "twitter_mention", "reddit_post"}
    if platform not in social:
        return None
    # Power-law distribution: mostly small, a few big
    base = np.random.pareto(1.5) * 500
    return int(min(base, 2_000_000))


def _star_rating(platform: str, sentiment: str) -> int | None:
    """Star ratings only for review platforms and surveys."""
    star_platforms = {
        "trustpilot_review", "google_review", "post_purchase_survey",
    }
    if platform not in star_platforms:
        return None
    if sentiment == "positive":
        return random.choices([4, 5], weights=[0.3, 0.7])[0]
    elif sentiment == "neutral":
        return random.choices([2, 3, 4], weights=[0.15, 0.65, 0.2])[0]
    else:
        return random.choices([1, 2], weights=[0.7, 0.3])[0]


def _generate_text(
    platform: str,
    sentiment: str,
    category: str,
    is_crisis: bool = False,
    return_reason: str | None = None,
) -> str:
    """Select and fill a template with a product name."""
    product = _pick_product_name(category)

    if is_crisis:
        template = random.choice(CRISIS_TEMPLATES)
        product = CRISIS_PRODUCT
    elif platform == "returns_feedback_form" and return_reason:
        templates = RETURN_FEEDBACK_TEMPLATES.get(return_reason, NEGATIVE_TEMPLATES)
        template = random.choice(templates)
    elif platform == "post_purchase_survey":
        templates = SURVEY_TEMPLATES.get(sentiment, SURVEY_TEMPLATES["neutral"])
        template = random.choice(templates)
    else:
        if sentiment == "positive":
            template = random.choice(POSITIVE_TEMPLATES)
        elif sentiment == "neutral":
            template = random.choice(NEUTRAL_TEMPLATES)
        else:
            template = random.choice(NEGATIVE_TEMPLATES)

    return template.format(product=product)


def generate_dataset(
    n_mentions: int = 12_000,
    start_date: str = "2025-07-01",
    months: int = 6,
    crisis_month: int = 4,
    crisis_count: int = 800,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Generate the full synthetic dataset.

    Parameters
    ----------
    n_mentions : int
        Total number of mentions (excluding crisis injection).
    start_date : str
        ISO start date.
    months : int
        Duration in months.
    crisis_month : int
        1-indexed month where crisis is injected.
    crisis_count : int
        Number of crisis mentions to inject.
    output_path : str or None
        If provided, save CSV to this path.

    Returns
    -------
    pd.DataFrame
    """
    start = datetime.fromisoformat(start_date)
    end = start + timedelta(days=30 * months)
    total_seconds = int((end - start).total_seconds())

    records: list[dict] = []

    # --- Normal mentions ---
    for i in range(n_mentions):
        ts = start + timedelta(seconds=random.randint(0, total_seconds))
        platform = random.choices(PLATFORMS, weights=PLATFORM_WEIGHTS)[0]
        category = random.choices(PRODUCT_CATEGORIES, weights=CATEGORY_WEIGHTS)[0]
        sku = random.choice(SKUS[category])
        region = random.choices(REGIONS, weights=REGION_WEIGHTS)[0]
        value_band = random.choices(ORDER_VALUE_BANDS, weights=VALUE_BAND_WEIGHTS)[0]

        # Sentiment distribution: 55% positive, 20% neutral, 25% negative
        sentiment = random.choices(
            ["positive", "neutral", "negative"], weights=[0.55, 0.20, 0.25]
        )[0]

        is_return = platform == "returns_feedback_form"
        return_reason = (
            random.choices(RETURN_REASONS, weights=RETURN_REASON_WEIGHTS)[0]
            if is_return
            else None
        )

        # Returns overweight negative sentiment
        if is_return:
            sentiment = random.choices(
                ["positive", "neutral", "negative"], weights=[0.05, 0.15, 0.80]
            )[0]

        verified = None
        if platform in ("trustpilot_review", "google_review"):
            verified = random.random() < 0.75

        text = _generate_text(platform, sentiment, category, return_reason=return_reason)
        followers = _generate_follower_count(platform)
        stars = _star_rating(platform, sentiment)

        records.append(
            {
                "mention_id": _generate_mention_id(i),
                "timestamp": ts,
                "platform": platform,
                "text": text,
                "product_category": category,
                "product_sku": sku,
                "star_rating": stars,
                "author_follower_count": followers,
                "verified_purchase": verified,
                "order_value_band": value_band,
                "region": region,
                "return_reason": return_reason,
            }
        )

    # --- Crisis injection in month 4 (800 mentions over 2 weeks) ---
    crisis_start = start + timedelta(days=30 * (crisis_month - 1))
    crisis_end = crisis_start + timedelta(days=14)
    crisis_seconds = int((crisis_end - crisis_start).total_seconds())

    # Heavy TikTok + Trustpilot skew for crisis
    crisis_platforms = [
        "tiktok_comment", "twitter_mention", "instagram_comment",
        "trustpilot_review", "reddit_post", "returns_feedback_form",
    ]
    crisis_platform_weights = [0.35, 0.20, 0.15, 0.12, 0.10, 0.08]

    for j in range(crisis_count):
        ts = crisis_start + timedelta(seconds=random.randint(0, crisis_seconds))
        platform = random.choices(crisis_platforms, weights=crisis_platform_weights)[0]
        region = random.choices(REGIONS, weights=REGION_WEIGHTS)[0]
        value_band = random.choices(ORDER_VALUE_BANDS, weights=VALUE_BAND_WEIGHTS)[0]

        is_return = platform == "returns_feedback_form"
        return_reason = "quality" if is_return else None

        text = _generate_text(
            platform, "negative", "apparel", is_crisis=True, return_reason=return_reason
        )
        followers = _generate_follower_count(platform)
        # Boost some follower counts for viral effect
        if followers and random.random() < 0.10:
            followers = random.randint(15_000, 500_000)

        stars = _star_rating(platform, "negative")
        verified = None
        if platform in ("trustpilot_review", "google_review"):
            verified = random.random() < 0.80

        records.append(
            {
                "mention_id": _generate_mention_id(n_mentions + j),
                "timestamp": ts,
                "platform": platform,
                "text": text,
                "product_category": "apparel",
                "product_sku": CRISIS_SKU,
                "star_rating": stars,
                "author_follower_count": followers,
                "verified_purchase": verified,
                "order_value_band": value_band,
                "region": region,
                "return_reason": return_reason,
            }
        )

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Enforce types
    df["star_rating"] = df["star_rating"].astype("Int64")
    df["author_follower_count"] = df["author_follower_count"].astype("Int64")
    df["verified_purchase"] = df["verified_purchase"].astype("boolean")

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"✅ Generated {len(df):,} mentions → {output_path}")

    return df


if __name__ == "__main__":
    import os

    out = os.path.join(os.path.dirname(__file__), "retail_mentions.csv")
    df = generate_dataset(output_path=out)

    # Quick summary stats
    print(f"\nDate range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"\nPlatform distribution:\n{df['platform'].value_counts()}")
    print(f"\nCategory distribution:\n{df['product_category'].value_counts()}")
    print(f"\nCrisis SKU mentions: {(df['product_sku'] == CRISIS_SKU).sum()}")
