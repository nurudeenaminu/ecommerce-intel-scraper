# E-Commerce Intelligence Scraper → RAG-Ready Pipeline

> Scrapes real Amazon product listings and reviews, cleans and structures the data, chunks it into ~500-token segments with full metadata, and loads it into a ChromaDB vector store — ready to power semantic search, AI chatbots, or any RAG application.

---

## What This Does (And Why It Matters)

Most data pipelines stop at collection. This one goes the full distance:

**Collection → Cleaning → Structuring → AI-Ready Indexing**

Any company building an AI product on top of product data — a shopping assistant, a review summarizer, a competitor intelligence tool, a recommendation engine — needs exactly this pipeline before they can write a single line of AI logic. This project delivers that foundation, end to end.

---

## Business Use Cases

| Use Case | How This Pipeline Enables It |
|---|---|
| AI shopping assistant | Query the vector store with natural language to surface relevant products |
| Review sentiment analysis | Clean, chunked review text is ready for any LLM or classifier |
| Competitor price monitoring | Structured pricing data with product metadata attached |
| Product recommendation engine | Semantic similarity search across descriptions and reviews |
| E-commerce RAG chatbot | Drop the JSONL directly into any vector store (Pinecone, Weaviate, Qdrant) |

---

## The Stack

| Layer | Tool | Purpose |
|---|---|---|
| Scraping | Playwright + playwright-stealth | Handles JavaScript-rendered pages, bypasses bot detection |
| Cleaning | pandas | Normalizes text, removes HTML artifacts, deduplicates |
| Chunking | tiktoken | Splits text into ~500-token chunks with 50-token overlap |
| Formatting | Python + json | Outputs JSONL with full metadata on every chunk |
| Vector Store | ChromaDB + sentence-transformers | Embeds and indexes chunks for semantic retrieval |

---

## Project Structure

```
ecommerce-intel-scraper/
├── scraper.py          # Playwright scraper — products, prices, reviews
├── cleaner.py          # pandas cleaning layer
├── chunker.py          # Token-aware chunker → JSONL formatter
├── loader.py           # ChromaDB ingestion + semantic query demo
├── data/
│   ├── raw/            # products_raw.json        (scraper output)
│   ├── cleaned/        # products_clean.csv       (cleaner output)
│   └── chunks/         # products_chunks.jsonl    (chunker output)
├── chroma_db/          # Persistent vector store (auto-created)
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/YOUR_USERNAME/ecommerce-intel-scraper.git
cd ecommerce-intel-scraper

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
playwright install chromium
```

### 2. Run the full pipeline

```bash
# Step 1 — Scrape Amazon (~15-30 min depending on pages/delays)
python scraper.py

# Step 2 — Clean and normalize
python cleaner.py

# Step 3 — Chunk and format to JSONL
python chunker.py

# Step 4 — Load into ChromaDB and run semantic queries
python loader.py
```

---

## Configuration

All key settings live at the top of each script — no config file needed:

**`scraper.py`**
```python
SEARCH_QUERY = "wireless headphones"   # Change to any product category
MAX_PAGES    = 3                       # Number of search result pages
```

**`chunker.py`**
```python
CHUNK_SIZE    = 500   # tokens per chunk
CHUNK_OVERLAP = 50    # overlap between consecutive chunks
```

---

## Output Format

Every line in `products_chunks.jsonl` is a self-contained JSON object:

```json
{
  "text": "Industry-leading noise cancellation automatically optimizes 
           for your environment. Up to 30-hour battery life with quick 
           charging — 10 minutes for 1.5 hours of playback...",
  "chunk_id": "a3f9c821-desc-0",
  "type": "description",
  "product_name": "Sony WH-1000XM5 Wireless Headphones",
  "price_usd": 279.99,
  "rating": "4.4 out of 5 stars",
  "category": "wireless headphones",
  "source_url": "https://www.amazon.com/dp/B09XS7JWHH"
}
```

Every chunk carries the full product context — so when a RAG system retrieves it, it knows exactly what product it came from, what it cost, and where to send the user.

---

## Semantic Query Results (Sample)

After loading 74 chunks from 24 Amazon products:

```
Query: 'best noise cancellation for flights'
  Result 1 — Sony WH-CH720N Noise Canceling Wireless Headphones
             Price: $99.99  |  Rating: 4.4 out of 5 stars
             Type: description
             "...Dual Noise Sensor technology captures noise from two
              directions for more precise cancellation in loud environments
              like airplanes and commutes..."

Query: 'battery life issues and complaints'
  Result 1 — Soundcore by Anker Q20i Hybrid Active Noise Cancelling
             Price: $39.99  |  Rating: 4.3 out of 5 stars
             Type: review
             "...battery drains much faster with ANC enabled, barely
              getting 15 hours instead of the advertised 40..."

Query: 'good headphones under 50 dollars'
  Result 1 — JBL Tune 720BT Wireless Over-Ear Headphones
             Price: $49.95  |  Rating: 4.4 out of 5 stars
             Type: description
             "...Pure Bass sound, 76-hour battery life, hands-free calls
              with the built-in microphone..."
```

---

## Extending This Pipeline

**Swap the vector store** — the JSONL output is format-agnostic. Replace ChromaDB with:
```python
# Pinecone
index.upsert(vectors=[(id, embedding, metadata)])

# Qdrant
client.upsert(collection_name="products", points=[...])

# FAISS
index.add(np.array(embeddings))
```

**Connect to an LLM** — pipe query results into any chat model:
```python
context = "\n\n".join([r for r in results["documents"][0]])
prompt  = f"Based on these products:\n{context}\n\nAnswer: {user_question}"
```

**Scale to more categories** — change `SEARCH_QUERY` and re-run. The pipeline handles any Amazon product category without modification.

---

## Requirements

```
playwright>=1.40.0
playwright-stealth
pandas>=2.0.0
tiktoken>=0.5.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
```

Install everything:
```bash
pip install -r requirements.txt
playwright install chromium
```

---

## Important Notes

- This project is for **educational and portfolio purposes**. Always review a site's `robots.txt` and Terms of Service before scraping in production.
- Randomized delays between requests (4–9 seconds) are built in to avoid overloading servers.
- Amazon selectors change over time. If the scraper returns 0 products, run `diagnose.py` to identify the current selectors.

---

## Author

Built as a portfolio project demonstrating the full data value chain:
**collection → cleaning → AI-ready structuring → semantic retrieval**

The kind of pipeline any company building on top of product data needs before they can write a single line of AI logic.