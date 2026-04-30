import json
import uuid
import pandas as pd
import tiktoken
from pathlib import Path

INPUT_FILE = "data/cleaned/products_clean.csv"
OUTPUT_FILE = "data/chunks/products_chunks.jsonl"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

enc = tiktoken.get_encoding("cl100k_base")

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text or not isinstance(text, str):
        return []
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens).strip())
        if end >= len(tokens):
            break
        start += chunk_size - overlap
    return [c for c in chunks if c]

def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Processing {len(df)} products into chunks...")

    Path("data/chunks").mkdir(parents=True, exist_ok=True)
    total_chunks = 0
    seen_ids = set()   # catches any remaining duplicates

    with open(OUTPUT_FILE, "w") as out:
        for _, row in df.iterrows():
            product_meta = {
                "product_name": row["title"],
                "price":    row.get("price"),
                "rating":       row.get("rating"),
                "category":     row.get("category"),
                "source_url":   row.get("source_url"),
            }

            # Description chunks
            if pd.notna(row.get("description")) and str(row["description"]).strip():
                for i, chunk in enumerate(chunk_text(str(row["description"]))):
                    chunk_id = f"{str(uuid.uuid4())}_desc_{i}"
                    if chunk_id in seen_ids:
                        continue
                    seen_ids.add(chunk_id)
                    out.write(json.dumps({
                        "text":     chunk,
                        "chunk_id": chunk_id,
                        "type":     "description",
                        **product_meta,
                    }) + "\n")
                    total_chunks += 1

            # Review chunks
            if pd.notna(row.get("reviews_text")) and str(row["reviews_text"]).strip():
                combined = " ".join(str(row["reviews_text"]).split(" ||| "))
                for i, chunk in enumerate(chunk_text(combined)):
                    chunk_id = f"{str(uuid.uuid4())}_review_{i}"
                    if chunk_id in seen_ids:
                        continue
                    seen_ids.add(chunk_id)
                    out.write(json.dumps({
                        "text":     chunk,
                        "chunk_id": chunk_id,
                        "type":     "review",
                        **product_meta,
                    }) + "\n")
                    total_chunks += 1

    print(f"Done. {total_chunks} chunks written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()