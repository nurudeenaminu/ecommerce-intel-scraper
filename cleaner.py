import json
import re
import pandas as pd
from pathlib import Path

INPUT_FILE = "data/raw/products_raw.json"
OUTPUT_FILE = "data/cleaned/products_clean.csv"

def clean_text(text):
    if not text or not isinstance(text, str):
        return None
    text = re.sub(r'<[^>]+>', '', text)        # strip HTML tags
    text = re.sub(r'\s+', ' ', text)           # collapse whitespace
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # remove non-ASCII
    return text.strip()

def clean_price(price):
    if not price:
        return None
    # Remove any currency symbol (GHS, $, £, €, etc.) and commas
    digits = re.sub(r'[^\d.]', '', str(price))
    try:
        return float(digits)
    except ValueError:
        return None

def clean_rating(rating):
    if not rating:
        return None
    match = re.search(r'[\d.]+', str(rating))
    return float(match.group()) if match else None

def flatten_reviews(reviews):
    if not reviews or not isinstance(reviews, list):
        return None
    cleaned = [clean_text(r) for r in reviews if r]
    cleaned = [r for r in cleaned if r and len(r) > 20]
    return " ||| ".join(cleaned)  # separator for splitting later

def main():
    with open(INPUT_FILE, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    print(f"Loaded {len(raw)} raw products")

    rows = []
    for item in raw:
        row = {
            "title":        clean_text(item.get("title")),
            "price":    clean_price(item.get("price")),
            "rating":       clean_rating(item.get("rating")),
            "review_count": item.get("review_count"),
            "description":  clean_text(item.get("description")),
            "reviews_text": flatten_reviews(item.get("reviews")),
            "category":     clean_text(item.get("category")),
            "source_url":   item.get("source_url"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Drop rows with no title or no meaningful content
    df = df.dropna(subset=["title"])
    df = df[df["title"].str.len() > 5]

    # Deduplicate by title
    df = df.drop_duplicates(subset=["title"], keep="first")

    # Fill missing prices with median
    median_price = df["price"].median()
    df["price"] = df["price"].fillna(median_price)

    # Reset index
    df = df.reset_index(drop=True)

    Path("data/cleaned").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Cleaned: {len(df)} unique products")
    print(f"Columns: {list(df.columns)}")
    print(f"Saved to {OUTPUT_FILE}")
    print("\nSample row:")
    print(df.iloc[0].to_dict())

if __name__ == "__main__":
    main()
