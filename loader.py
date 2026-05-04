import json
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

JSONL_FILE       = "data/chunks/products_chunks.jsonl"
COLLECTION_NAME  = "amazon_products"
TOP_K            = 5   # results to return per query

# ─── LOAD CHUNKS ─────────────────────────────────────────
def load_chunks():
    chunks = []
    with open(JSONL_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks

# ─── BUILD CHROMADB COLLECTION ───────────────────────────
def build_collection(client, chunks, model):
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    ids       = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{
        "product_name":    str(c.get("product_name", "")),
        "price_usd":       str(c.get("price_usd", "0")),
        "rating":          str(c.get("rating", "0")),
        "sentiment_label": str(c.get("sentiment_label", "")),
        "sentiment_score": str(c.get("sentiment_score", "0")),
        "category":        str(c.get("category", "")),
        "type":            str(c.get("type", "")),
        "source_url":      str(c.get("source_url", "")),
    } for c in chunks]

    print(f"Embedding {len(documents)} chunks...")
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
        )
        print(f"  Inserted {min(i+batch_size, len(chunks))}/{len(chunks)}")

    print(f"Collection ready — {collection.count()} chunks indexed.")
    return collection

# ─── BUILD BM25 INDEX ────────────────────────────────────
def build_bm25(chunks):
    """Keyword search index — catches exact product names and model numbers."""
    tokenized = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized)

# ─── HYBRID SEARCH ───────────────────────────────────────
def hybrid_search(
    query,
    collection,
    model,
    chunks,
    bm25,
    top_k=TOP_K,
    category_filter=None,
    min_rating=None,
    max_price=None,
    sentiment_filter=None,
):
    """
    Combines:
      - Semantic search (ChromaDB embeddings)
      - Keyword search (BM25)
    Then applies metadata filters:
      - category, min_rating, max_price, sentiment_label
    """

    # ── Step 1: semantic search ──
    query_embedding = model.encode([query]).tolist()

    chroma_filters = {}
    conditions     = []

    if category_filter:
        conditions.append({"category": {"$eq": category_filter}})
    if sentiment_filter:
        conditions.append({"sentiment_label": {"$eq": sentiment_filter}})

    chroma_where = None
    if len(conditions) == 1:
        chroma_where = conditions[0]
    elif len(conditions) > 1:
        chroma_where = {"$and": conditions}

    semantic_results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k * 3, collection.count()),
        where=chroma_where if chroma_where else None,
    )
    semantic_ids = semantic_results["ids"][0]

    # ── Step 2: BM25 keyword search ──
    tokenized_query = query.lower().split()
    bm25_scores     = bm25.get_scores(tokenized_query)
    top_bm25_idx    = np.argsort(bm25_scores)[::-1][:top_k * 3]
    bm25_ids        = {chunks[i]["chunk_id"] for i in top_bm25_idx}

    # ── Step 3: merge — union of both result sets ──
    combined_ids = list(dict.fromkeys(semantic_ids + list(bm25_ids)))

    # ── Step 4: retrieve full chunk objects ──
    chunk_map     = {c["chunk_id"]: c for c in chunks}
    candidates    = [chunk_map[cid] for cid in combined_ids if cid in chunk_map]

    # ── Step 5: metadata filtering ──
    filtered = []
    for c in candidates:
        if min_rating is not None:
            try:
                if float(c.get("rating", 0)) < min_rating:
                    continue
            except (ValueError, TypeError):
                pass
        if max_price is not None:
            try:
                if float(c.get("price_usd", 0)) > max_price:
                    continue
            except (ValueError, TypeError):
                pass
        filtered.append(c)

    return filtered[:top_k]

# ─── DEMO QUERIES ────────────────────────────────────────
def run_demo_queries(collection, model, chunks, bm25):
    demo_queries = [
        {
            "query":      "best noise cancellation for flights",
            "max_price":  None,
            "min_rating": 4.0,
        },
        {
            "query":           "budget headphones under 50 dollars",
            "max_price":       50.0,
            "min_rating":      None,
            "sentiment_filter": "positive",
        },
        {
            "query":      "battery life issues",
            "max_price":  None,
            "min_rating": None,
            "sentiment_filter": "negative",
        },
        {
            "query":           "wireless keyboard for mac",
            "category_filter": "wireless keyboards",
            "max_price":       None,
            "min_rating":      4.0,
        },
    ]

    print("\n" + "="*60)
    print("  HYBRID SEARCH — DEMO QUERY RESULTS")
    print("="*60)

    for q in demo_queries:
        print(f"\n  Query   : '{q['query']}'")
        filters = {k: v for k, v in q.items() if k != "query" and v is not None}
        if filters:
            print(f"  Filters : {filters}")
        print("  " + "-"*50)

        results = hybrid_search(
            query=q["query"],
            collection=collection,
            model=model,
            chunks=chunks,
            bm25=bm25,
            category_filter=q.get("category_filter"),
            min_rating=q.get("min_rating"),
            max_price=q.get("max_price"),
            sentiment_filter=q.get("sentiment_filter"),
        )

        if not results:
            print("  No results matched the filters.")
            continue

        for i, r in enumerate(results[:3]):
            print(f"\n  [{i+1}] {r.get('product_name', 'N/A')[:55]}")
            print(f"       Price: USD {r.get('price_usd', 'N/A')}  |  "
                  f"Rating: {r.get('rating', 'N/A')}  |  "
                  f"Sentiment: {r.get('sentiment_label', 'N/A')} "
                  f"({r.get('sentiment_score', 'N/A')})")
            print(f"       {r.get('text', '')[:160]}...")

    # Save results
    print(f"\n  Results saved to query_results.txt")
    with open("query_results.txt", "w", encoding="utf-8") as f:
        f.write("AMAZON RAG PIPELINE — HYBRID SEARCH RESULTS\n")
        f.write(f"Chunks indexed: {collection.count()} | "
                f"Products: 316 | Categories: 10\n\n")
        for q in demo_queries:
            f.write(f"Query: {q['query']}\n")
            results = hybrid_search(
                query=q["query"],
                collection=collection,
                model=model,
                chunks=chunks,
                bm25=bm25,
                category_filter=q.get("category_filter"),
                min_rating=q.get("min_rating"),
                max_price=q.get("max_price"),
                sentiment_filter=q.get("sentiment_filter"),
            )
            for i, r in enumerate(results[:3]):
                f.write(f"  [{i+1}] {r.get('product_name', 'N/A')}\n")
                f.write(f"       USD {r.get('price_usd', 'N/A')} | "
                        f"Rating {r.get('rating', 'N/A')} | "
                        f"Sentiment: {r.get('sentiment_label', 'N/A')}\n")
# ─── MAIN ────────────────────────────────────────────────
def main():
    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Found {len(chunks)} chunks\n")

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model ready.\n")

    client = chromadb.PersistentClient(path="./chroma_db")

    # Wipe old db if exists
    try:
        import shutil
        shutil.rmtree("./chroma_db")
        print("Cleared old chroma_db.\n")
    except Exception:
        pass

    client = chromadb.PersistentClient(path="./chroma_db")

    print("Building ChromaDB collection...")
    collection = build_collection(client, chunks, model)

    print("\nBuilding BM25 keyword index...")
    bm25 = build_bm25(chunks)
    print("BM25 ready.\n")

    run_demo_queries(collection, model, chunks, bm25)

if __name__ == "__main__":
    main()