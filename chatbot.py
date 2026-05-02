import json
import os
import re
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from google import genai
from google.genai import types

# ─── CONFIG ───────────────────────────────────────────────
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "amazon_products"
JSONL_FILE      = "data/chunks/products_chunks.jsonl"
TOP_K           = 5
BM25_WEIGHT     = 0.3

# ─── SYSTEM PROMPT ────────────────────────────────────────
SYSTEM_PROMPT = """You are a direct, helpful product assistant with access to real Amazon product data.

Rules:
- Be concise. 2-4 sentences max per recommendation unless asked for more detail
- Talk like a knowledgeable friend, not a customer service rep or report writer
- Lead with the answer, then the reason. Never start with "Certainly!" or "Great question!"
- When recommending products: name it, price it, one-line why it fits
- If multiple products match, use a short bullet list — product name, price, one reason
- If you don't have the data, say so in one sentence
- Never repeat the user's question back to them
- No intros, no summaries, no "I hope this helps" endings"""

# ─── LOAD RESOURCES ───────────────────────────────────────
def load_chunks():
    chunks = []
    with open(JSONL_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks

def build_bm25(chunks):
    tokenized = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized)

def load_chroma():
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    return collection

def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def load_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not set.\n"
            "Run: $env:GEMINI_API_KEY = 'your-key-here'"
        )
    client = genai.Client(api_key=api_key)
    return client

# ─── FILTER PARSING ───────────────────────────────────────
def parse_filters(user_message):
    msg     = user_message.lower()
    filters = {}

    # Max price
    price_max = re.search(
        r'under\s+\$?(\d+)|less\s+than\s+\$?(\d+)|below\s+\$?(\d+)|\$?(\d+)\s+or\s+less',
        msg
    )
    if price_max:
        val = next(v for v in price_max.groups() if v is not None)
        filters["max_price"] = float(val)

    # Min price
    price_min = re.search(
        r'over\s+\$?(\d+)|more\s+than\s+\$?(\d+)|above\s+\$?(\d+)',
        msg
    )
    if price_min:
        val = next(v for v in price_min.groups() if v is not None)
        filters["min_price"] = float(val)

    # Rating
    if any(w in msg for w in ["highly rated", "top rated", "best rated", "4 star", "5 star"]):
        filters["min_rating"] = 4.0

    # Sentiment
    if any(w in msg for w in ["complaint", "problem", "issue", "bad review", "negative"]):
        filters["sentiment"] = "negative"
    elif any(w in msg for w in ["positive review", "recommended", "loved", "great review"]):
        filters["sentiment"] = "positive"

    # Category
    category_map = {
        "headphone":    "wireless headphones",
        "speaker":      "bluetooth speakers",
        "earbud":       "wireless earbuds",
        "noise cancel": "noise cancelling headphones",
        "gaming":       "gaming headsets",
        "watch":        "smart watches",
        "keyboard":     "wireless keyboards",
        "webcam":       "webcams for streaming",
        "charger":      "portable chargers",
        "laptop stand": "laptop stands",
    }
    for keyword, category in category_map.items():
        if keyword in msg:
            filters["category"] = category
            break

    return filters

# ─── HYBRID SEARCH ────────────────────────────────────────
def hybrid_search(query, collection, embedder, bm25, all_chunks, filters, top_k=TOP_K):
    # ── Semantic search ──
    query_embedding = embedder.encode([query]).tolist()

    where_clause = {}
    if "category" in filters:
        where_clause["category"] = {"$eq": filters["category"]}
    if "sentiment" in filters:
        where_clause["sentiment_label"] = {"$eq": filters["sentiment"]}
    if "min_rating" in filters:
        where_clause["rating"] = {"$gte": filters["min_rating"]}

    chroma_kwargs = {
        "query_embeddings": query_embedding,
        "n_results":        min(top_k * 3, collection.count()),
        "include":          ["documents", "metadatas", "distances"],
    }
    if where_clause:
        chroma_kwargs["where"] = where_clause

    try:
        sem_results = collection.query(**chroma_kwargs)
    except Exception:
        sem_results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k * 3, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

    sem_docs  = sem_results["documents"][0]
    sem_metas = sem_results["metadatas"][0]
    sem_dists = sem_results["distances"][0]

    max_dist   = max(sem_dists) if sem_dists else 1
    sem_scores = {
        doc: 1 - (dist / max_dist)
        for doc, dist in zip(sem_docs, sem_dists)
    }

    # ── BM25 keyword search ──
    tokenized_query  = query.lower().split()
    bm25_scores_raw  = bm25.get_scores(tokenized_query)
    max_bm25         = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1
    bm25_norm        = [s / max_bm25 for s in bm25_scores_raw]

    top_bm25_indices = sorted(
        range(len(bm25_norm)), key=lambda i: bm25_norm[i], reverse=True
    )[:top_k * 3]

    # ── Combine scores ──
    combined = {}
    for doc, meta, sem_score in zip(sem_docs, sem_metas, sem_scores.values()):
        combined[doc] = {"meta": meta, "score": (1 - BM25_WEIGHT) * sem_score}

    for idx in top_bm25_indices:
        doc  = all_chunks[idx]["text"]
        meta = {k: str(v) for k, v in all_chunks[idx].items() if k != "text"}
        if doc in combined:
            combined[doc]["score"] += BM25_WEIGHT * bm25_norm[idx]
        else:
            combined[doc] = {"meta": meta, "score": BM25_WEIGHT * bm25_norm[idx]}

    # ── Price post-filter ──
    filtered = {}
    for doc, data in combined.items():
        price = float(data["meta"].get("price_usd", 0) or 0)
        if "max_price" in filters and price > 0 and price > filters["max_price"]:
            continue
        if "min_price" in filters and price > 0 and price < filters["min_price"]:
            continue
        filtered[doc] = data

    if not filtered:
        filtered = combined

    ranked = sorted(filtered.items(), key=lambda x: x[1]["score"], reverse=True)
    return ranked[:top_k]

# ─── CONTEXT BUILDER ──────────────────────────────────────
def build_context(results):
    context_parts = []
    seen_products = set()

    for doc, data in results:
        meta         = data["meta"]
        product_name = meta.get("product_name", "Unknown")
        price        = meta.get("price_usd", "0")
        rating       = meta.get("rating", "N/A")
        sentiment    = meta.get("sentiment_label", "N/A")
        category     = meta.get("category", "N/A")
        url          = meta.get("source_url", "")
        chunk_type   = meta.get("type", "")

        header = ""
        if product_name not in seen_products:
            seen_products.add(product_name)
            try:
                price_str = f"USD {float(price):.2f}" if float(price) > 0 else "See site"
            except Exception:
                price_str = "See site"
            header = (
                f"Product: {product_name}\n"
                f"Price: {price_str} | Rating: {rating}/5 | "
                f"Sentiment: {sentiment} | Category: {category}\n"
                f"URL: {url}\n"
            )

        context_parts.append(f"{header}[{chunk_type.upper()}]: {doc}\n")

    return "\n---\n".join(context_parts)

# ─── CHAT FUNCTION ────────────────────────────────────────
def chat(user_message, history, collection, embedder, bm25, all_chunks, gemini_client):
    # Parse filters
    filters = parse_filters(user_message)
    if filters:
        print(f"  [Filters: {filters}]")

    # Retrieve chunks
    results = hybrid_search(
        user_message, collection, embedder, bm25, all_chunks, filters
    )

    context = build_context(results) if results else "No relevant products found."

    # Build message history for Gemini
    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(
            role=role,
            parts=[types.Part(text=msg["content"])]
        ))

    # Add current message with retrieved context
    full_message = (
        f"Product data from our database:\n\n{context}\n\n"
        f"---\nUser question: {user_message}"
    )
    contents.append(types.Content(
        role="user",
        parts=[types.Part(text=full_message)]
    ))

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1024,
        ),
        contents=contents,
    )
    assistant_reply = response.text

    # Update history with clean versions — no context bloat
    history.append({"role": "user",      "content": user_message})
    history.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply, history

# ─── MAIN CLI LOOP ────────────────────────────────────────
def main():
    print("Loading pipeline...")
    all_chunks   = load_chunks()
    collection   = load_chroma()
    embedder     = load_embedder()
    bm25         = build_bm25(all_chunks)
    gemini_client = load_gemini()

    print(f"\n{'='*58}")
    print(f"  E-Commerce RAG Chatbot  (Powered by Gemini 2.5 Flash)")
    print(f"  {collection.count()} chunks | 10 categories | Hybrid search")
    print(f"  Commands: 'quit' to exit | 'clear' to reset memory")
    print(f"{'='*58}\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye.")
            break
        if user_input.lower() == "clear":
            history = []
            print("  Memory cleared.\n")
            continue

        try:
            response, history = chat(
                user_input, history, collection,
                embedder, bm25, all_chunks, gemini_client
            )
            print(f"\nAssistant: {response}\n")
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and "retryDelay" in error_str:
                # Extract wait time from error
                import re
                delay_match = re.search(r"retryDelay.*?(\d+)s", error_str)
                wait = int(delay_match.group(1)) + 2 if delay_match else 30
                print(f"\n  Rate limited. Waiting {wait}s then retrying...\n")
                import time
                time.sleep(wait)
                try:
                    response, history = chat(
                        user_input, history, collection,
                        embedder, bm25, all_chunks, gemini_client
                    )
                    print(f"\nAssistant: {response}\n")
                except Exception as e2:
                    print(f"\n  Still failing: {e2}\n")
            else:
                print(f"\n  Error: {e}\n")

if __name__ == "__main__":
    main()