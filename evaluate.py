import json
import os
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
import time

# ─── CONFIG ───────────────────────────────────────────────
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "amazon_products"
JSONL_FILE      = "data/chunks/products_chunks.jsonl"
TOP_K           = 5
BM25_WEIGHT     = 0.3
RESULTS_FILE    = "eval_results.json"

SYSTEM_PROMPT = """You are a knowledgeable e-commerce product assistant.
Answer based ONLY on the product data provided. Be concise and factual."""

EVAL_SET = [
    {
        "id": 1,
        "question": "What Sony noise cancelling headphones do you have?",
        "expected_keywords": ["sony", "noise cancel"],
        "category": "product lookup",
    },
    {
        "id": 2,
        "question": "Show me wireless earbuds under $50",
        "expected_keywords": ["earbud", "dollar", "50"],
        "category": "price filter",
    },
    {
        "id": 3,
        "question": "What are customers complaining about with gaming headsets?",
        "expected_keywords": ["gaming", "sound", "comfort"],
        "category": "sentiment — negative",
    },
    {
        "id": 4,
        "question": "Which portable charger has the best reviews?",
        "expected_keywords": ["charger", "rating", "review"],
        "category": "product rating",
    },
    {
        "id": 5,
        "question": "Tell me about the battery life of wireless headphones",
        "expected_keywords": ["battery", "hour", "life"],
        "category": "feature query",
    },
    {
        "id": 6,
        "question": "What webcams do you have for streaming?",
        "expected_keywords": ["webcam", "stream", "camera"],
        "category": "category filter",
    },
    {
        "id": 7,
        "question": "Are there any highly rated smart watches?",
        "expected_keywords": ["watch", "rating", "star"],
        "category": "rating filter",
    },
    {
        "id": 8,
        "question": "What do customers love about bluetooth speakers?",
        "expected_keywords": ["speaker", "sound", "bluetooth"],
        "category": "sentiment — positive",
    },
    {
        "id": 9,
        "question": "Do you have any wireless keyboards for mac?",
        "expected_keywords": ["keyboard", "wireless", "mac"],
        "category": "category + use case",
    },
    {
        "id": 10,
        "question": "What is the cheapest laptop stand in your data?",
        "expected_keywords": ["stand", "laptop", "price"],
        "category": "price ranking",
    },
]

# ─── LOAD RESOURCES ───────────────────────────────────────
def load_resources():
    chunks = []
    with open(JSONL_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    embedder   = SentenceTransformer("all-MiniLM-L6-v2")
    tokenized  = [c["text"].lower().split() for c in chunks]
    bm25       = BM25Okapi(tokenized)

    api_key    = os.environ.get("GEMINI_API_KEY", "")
    gemini     = genai.Client(api_key=api_key)

    return chunks, collection, embedder, bm25, gemini

# ─── HYBRID SEARCH ────────────────────────────────────────
def hybrid_search(query, collection, embedder, bm25, all_chunks, top_k=TOP_K):
    query_embedding = embedder.encode([query]).tolist()
    sem_results     = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k * 3, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    sem_docs  = sem_results["documents"][0]
    sem_metas = sem_results["metadatas"][0]
    sem_dists = sem_results["distances"][0]

    max_dist   = max(sem_dists) if sem_dists else 1
    sem_scores = {doc: 1 - (dist / max_dist)
                  for doc, dist in zip(sem_docs, sem_dists)}

    bm25_raw         = bm25.get_scores(query.lower().split())
    max_bm25         = max(bm25_raw) if max(bm25_raw) > 0 else 1
    bm25_norm        = [s / max_bm25 for s in bm25_raw]
    top_bm25_indices = sorted(
        range(len(bm25_norm)), key=lambda i: bm25_norm[i], reverse=True
    )[:top_k * 3]

    combined = {}
    for doc, meta, score in zip(sem_docs, sem_metas, sem_scores.values()):
        combined[doc] = {"meta": meta, "score": (1 - BM25_WEIGHT) * score}
    for idx in top_bm25_indices:
        doc  = all_chunks[idx]["text"]
        meta = {k: str(v) for k, v in all_chunks[idx].items() if k != "text"}
        if doc in combined:
            combined[doc]["score"] += BM25_WEIGHT * bm25_norm[idx]
        else:
            combined[doc] = {"meta": meta, "score": BM25_WEIGHT * bm25_norm[idx]}

    ranked = sorted(combined.items(), key=lambda x: x[1]["score"], reverse=True)
    return ranked[:top_k]

def build_context(results):
    parts         = []
    seen_products = set()
    for doc, data in results:
        meta         = data["meta"]
        product_name = meta.get("product_name", "Unknown")
        price        = meta.get("price_usd", "0")
        rating       = meta.get("rating", "N/A")
        sentiment    = meta.get("sentiment_label", "N/A")
        url          = meta.get("source_url", "")
        chunk_type   = meta.get("type", "")
        header       = ""
        if product_name not in seen_products:
            seen_products.add(product_name)
            try:
                price_str = f"USD {float(price):.2f}" if float(price) > 0 else "See site"
            except Exception:
                price_str = "See site"
            header = (
                f"Product: {product_name}\n"
                f"Price: {price_str} | Rating: {rating}/5 | "
                f"Sentiment: {sentiment}\nURL: {url}\n"
            )
        parts.append(f"{header}[{chunk_type.upper()}]: {doc}\n")
    return "\n---\n".join(parts)

# ─── SCORING ──────────────────────────────────────────────
def score_response(response_text, expected_keywords):
    response_lower = response_text.lower()
    matched        = [kw for kw in expected_keywords if kw.lower() in response_lower]
    score          = len(matched) / len(expected_keywords)
    return round(score, 2), matched

# ─── MAIN ─────────────────────────────────────────────────
def main():
    print("Loading pipeline for evaluation...")
    all_chunks, collection, embedder, bm25, gemini = load_resources()

    print(f"\n{'='*60}")
    print(f"  RAG PIPELINE EVALUATION — {len(EVAL_SET)} TEST QUESTIONS")
    print(f"  Chunks indexed: {collection.count()}")
    print(f"{'='*60}\n")

    results     = []
    total_score = 0
    passed      = 0

    for item in EVAL_SET:
        print(f"  [{item['id']:02d}] {item['question'][:55]}...")
        
        # Wait between calls to respect free tier limits
        # Free tier: 5 req/min, 20 req/day
        # We wait 65 seconds between each to stay safe
        if item["id"] > 1:
            print(f"        Waiting 65s for rate limit...")
            time.sleep(65)

        search_results = hybrid_search(
            item["question"], collection, embedder, bm25, all_chunks
        )
        context = build_context(search_results) if search_results else "No data found."

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Product data:\n\n{context}\n\n"
            f"---\nQuestion: {item['question']}"
        )

        try:
            response = gemini.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            answer = response.text or ""
        except Exception as e:
            answer = f"ERROR: {e}"
            print(f"        API error: {e}")

        score, matched_kws = score_response(answer, item["expected_keywords"])
        passed_test        = score >= 0.5
        total_score       += score
        if passed_test:
            passed += 1

        missing = [k for k in item["expected_keywords"] if k not in matched_kws]
        status  = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"        Score: {score:.0%}  {status}")
        print(f"        Matched : {matched_kws}")
        print(f"        Missing : {missing}")
        print(f"        Answer  : {answer[:120]}...\n")

        results.append({
            "id":                item["id"],
            "question":          item["question"],
            "category":          item["category"],
            "expected_keywords": item["expected_keywords"],
            "matched_keywords":  matched_kws,
            "score":             score,
            "passed":            passed_test,
            "answer_preview":    answer[:300],
        })

    avg_score = total_score / len(EVAL_SET)
    pass_rate = passed / len(EVAL_SET)

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"  Questions tested : {len(EVAL_SET)}")
    print(f"  Passed (>=50%)   : {passed}/{len(EVAL_SET)}")
    print(f"  Pass rate        : {pass_rate:.0%}")
    print(f"  Average score    : {avg_score:.0%}")
    print(f"{'='*60}")
    print(f"\n  Results by category:")
    categories = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r["score"])
    for cat, scores in sorted(categories.items()):
        avg = sum(scores) / len(scores)
        bar = "█" * int(avg * 10) + "░" * (10 - int(avg * 10))
        print(f"  {cat:<25} {bar} {avg:.0%}")

    summary = {
        "total_questions": len(EVAL_SET),
        "passed":          passed,
        "pass_rate":       f"{pass_rate:.0%}",
        "average_score":   f"{avg_score:.0%}",
        "per_question":    results,
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()