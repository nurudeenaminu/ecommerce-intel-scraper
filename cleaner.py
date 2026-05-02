import json
import re
import pandas as pd
from pathlib import Path
from textblob import TextBlob

INPUT_FILE  = "data/raw/products_raw.json"
OUTPUT_FILE = "data/cleaned/products_clean.csv"

# ─── TEXT CLEANING ────────────────────────────────────────
def clean_text(text):
    if not text or not isinstance(text, str):
        return None
    text = re.sub(r'<[^>]+>', '', text)           # strip HTML tags
    text = re.sub(r'&[a-zA-Z]+;', '', text)       # strip HTML entities (&amp; etc)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)    # replace non-ASCII with space
    text = re.sub(r'\s+', ' ', text)              # collapse whitespace
    return text.strip() or None

def clean_price(price):
    if not price or not isinstance(price, str):
        return None
    # Remove ALL currency symbols and text (GHS, USD, $, £, etc.)
    digits = re.sub(r'[^\d.,]', '', price)
    # Handle comma-formatted numbers like 3,080.00
    digits = digits.replace(',', '')
    try:
        val = float(digits)
        return round(val, 2) if val > 0 else None
    except ValueError:
        return None
    
def clean_rating(rating):
    if not rating:
        return None
    match = re.search(r'[\d.]+', str(rating))
    return float(match.group()) if match else None

def clean_review_count(rc):
    if not rc:
        return None
    digits = re.sub(r'[^\d]', '', str(rc))
    return int(digits) if digits else None

# ─── SPAM REVIEW DETECTION ───────────────────────────────
def is_spam_review(review_text):
    """
    Flag a review as spam using simple heuristics:
    - Under 10 words (too short to be useful)
    - More than 50% repeated words (suspicious repetition)
    """
    if not review_text or not isinstance(review_text, str):
        return True
    words = review_text.lower().split()
    if len(words) < 10:
        return True
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.5:
        return True
    return False

def filter_reviews(reviews):
    """Remove spam reviews, return clean list."""
    if not reviews or not isinstance(reviews, list):
        return []
    cleaned = []
    for r in reviews:
        text = clean_text(r)
        if text and not is_spam_review(text):
            cleaned.append(text)
    return cleaned

# ─── SENTIMENT SCORING ───────────────────────────────────
def compute_sentiment(reviews):
    """
    Run TextBlob sentiment on combined review text.
    Returns:
        sentiment_score     : float -1.0 (negative) to +1.0 (positive)
        sentiment_label     : 'positive' | 'neutral' | 'negative'
        sentiment_subjectivity : float 0.0 (objective) to 1.0 (subjective)
    """
    if not reviews:
        return 0.0, "neutral", 0.0

    combined = " ".join(reviews)
    if not combined.strip():
        return 0.0, "neutral", 0.0

    blob = TextBlob(combined)
    score = round(blob.sentiment.polarity, 4)
    subjectivity = round(blob.sentiment.subjectivity, 4)

    if score > 0.1:
        label = "positive"
    elif score < -0.1:
        label = "negative"
    else:
        label = "neutral"

    return score, label, subjectivity

# ─── MAIN ─────────────────────────────────────────────────
def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    print(f"Loaded {len(raw)} raw products")

    rows = []
    seen_asins  = set()   # deduplicate by ASIN
    seen_titles = set()   # fallback dedup by title

    for item in raw:
        # ── ASIN deduplication ──
        asin = item.get("asin")
        if asin and asin in seen_asins:
            continue
        if asin:
            seen_asins.add(asin)

        title = clean_text(item.get("title"))
        if not title or len(title) < 5:
            continue

        # ── Title fallback dedup ──
        title_key = title.lower().strip()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)

        # ── Clean fields ──
        price        = clean_price(item.get("price"))
        rating       = clean_rating(item.get("rating"))
        review_count = clean_review_count(item.get("review_count"))
        description  = clean_text(item.get("description"))
        category     = clean_text(item.get("category"))
        source_url   = item.get("source_url", "")

        # ── Filter spam reviews ──
        raw_reviews     = item.get("reviews", [])
        clean_reviews   = filter_reviews(raw_reviews)
        spam_removed    = len(raw_reviews) - len(clean_reviews)

        # ── Sentiment scoring ──
        sentiment_score, sentiment_label, sentiment_subjectivity = compute_sentiment(clean_reviews)

        # ── Join reviews for storage ──
        reviews_text = " ||| ".join(clean_reviews) if clean_reviews else None

        rows.append({
            "asin":                    asin,
            "title":                   title,
            "price_usd":               price,
            "rating":                  rating,
            "review_count":            review_count,
            "description":             description,
            "reviews_text":            reviews_text,
            "sentiment_score":         sentiment_score,
            "sentiment_label":         sentiment_label,
            "sentiment_subjectivity":  sentiment_subjectivity,
            "spam_reviews_removed":    spam_removed,
            "category":                category,
            "source_url":              source_url,
        })

    df = pd.DataFrame(rows)

    # ── Fill missing prices with category median ──
    df["price_usd"] = df.groupby("category")["price_usd"].transform(
        lambda x: x.fillna(x.median())
    )
    # Global median fallback if category median is also NaN
    df["price_usd"] = df["price_usd"].fillna(df["price_usd"].median())

    df = df.reset_index(drop=True)

    Path("data/cleaned").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    # ── Summary ──
    print(f"\n{'='*55}")
    print(f"  CLEANING COMPLETE")
    print(f"  Input products     : {len(raw)}")
    print(f"  After dedup        : {len(df)}")
    print(f"  Duplicates removed : {len(raw) - len(df)}")
    print(f"  Columns            : {list(df.columns)}")
    print(f"\n  Sentiment breakdown:")
    print(df["sentiment_label"].value_counts().to_string())
    print(f"\n  Category breakdown:")
    print(df["category"].value_counts().to_string())
    print(f"\n  Price stats (USD):")
    print(df["price_usd"].describe().round(2).to_string())
    print(f"\n  Output: {OUTPUT_FILE}")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()