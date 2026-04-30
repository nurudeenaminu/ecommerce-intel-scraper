import json
import chromadb
from sentence_transformers import SentenceTransformer

JSONL_FILE = "data/chunks/products_chunks.jsonl"
COLLECTION_NAME = "amazon_products"

def load_chunks():
    chunks = []
    with open(JSONL_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks

def build_collection(client, chunks):
    print("Loading embedding model (downloads once, then cached)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model ready.")

    # Delete and recreate collection for clean re-runs
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    ids       = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "product_name": str(c.get("product_name", "")),
            "price":        str(c.get("price", "")),
            "rating":       str(c.get("rating", "")),
            "category":     str(c.get("category", "")),
            "type":         str(c.get("type", "")),
            "source_url":   str(c.get("source_url", "")),
        }
        for c in chunks
    ]

    # Generate embeddings in one batch
    print(f"Embedding {len(documents)} chunks...")
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    # Insert in batches of 50
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
        )
        print(f"  Inserted {min(i+batch_size, len(chunks))}/{len(chunks)} chunks")

    print(f"\nCollection '{COLLECTION_NAME}' ready — {collection.count()} chunks indexed.")
    return collection, model

def run_queries(collection, model):
    queries = [
        "best noise cancellation for flights",
        "battery life issues and complaints",
        "good headphones under 50 dollars",
    ]

    print("\n" + "="*60)
    print("SEMANTIC QUERY RESULTS")
    print("="*60)

    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)

        query_embedding = model.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3,
        )

        for i, (doc, meta) in enumerate(
            zip(results["documents"][0], results["metadatas"][0])
        ):
            print(f"\n  Result {i+1}:")
            print(f"  Product : {meta['product_name'][:60]}")
            print(f"  Price   : {meta['price']}  |  Rating: {meta['rating']}")
            print(f"  Type    : {meta['type']}")
            print(f"  Excerpt : {doc[:200]}...")

def main():
    print("Loading chunks from JSONL...")
    chunks = load_chunks()
    print(f"Found {len(chunks)} chunks\n")

    client = chromadb.PersistentClient(path="./chroma_db")

    print("Building ChromaDB collection...")
    collection, model = build_collection(client, chunks)

    run_queries(collection, model)

if __name__ == "__main__":
    main()