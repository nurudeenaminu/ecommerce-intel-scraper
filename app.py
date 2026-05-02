import json
import os
import re
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import google.genai as genai
from google.genai import types

# ─── CONFIG ───────────────────────────────────────────────
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "amazon_products"
JSONL_FILE      = "data/chunks/products_chunks.jsonl"
TOP_K           = 5
BM25_WEIGHT     = 0.3

SYSTEM_PROMPT = """You are a direct, helpful product assistant with access to real Amazon product data.

Rules:
- Be concise. 
- Talk like a knowledgeable friend, not a customer service rep or report writer
- Lead with the answer, then the reason. Never start with "Certainly!" or "Great question!"
- When recommending products: name it, price it, one-line why it fits
- If multiple products match, use a short bullet list — product name, price, one reason
- If you don't have the data, say so in one sentence
- Never repeat the user's question back to them
- No intros, no summaries, no "I hope this helps" endings"""

# ─── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce AI Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────
st.markdown("""
<style>
    .main { padding-top: 1rem; }

    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }

    .product-card {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 13px;
    }

    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 500;
        margin-right: 4px;
    }

    .badge-positive  { background:#d4edda; color:#155724; }
    .badge-negative  { background:#f8d7da; color:#721c24; }
    .badge-neutral   { background:#e2e3e5; color:#383d41; }
    .badge-price     { background:#cce5ff; color:#004085; }
    .badge-rating    { background:#fff3cd; color:#856404; }

    .filter-box {
        background: #e8f4f8;
        border-left: 3px solid #17a2b8;
        border-radius: 4px;
        padding: 6px 12px;
        font-size: 12px;
        color: #0c5460;
        margin-bottom: 8px;
    }

    .stat-box {
        text-align: center;
        padding: 12px;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }

    .stat-number { font-size: 24px; font-weight: 600; color: #1a1a2e; }
    .stat-label  { font-size: 11px; color: #6c757d; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ─── CACHED RESOURCE LOADERS ──────────────────────────────
@st.cache_resource
def load_all_resources():
    # Chunks
    chunks = []
    with open(JSONL_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    # ChromaDB — rebuild if collection missing or corrupt
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(COLLECTION_NAME)
        # Verify it has data
        if collection.count() == 0:
            raise ValueError("Empty collection")
    except Exception:
        st.info("Building search index for first time... (~2 min)")
        # Delete and recreate
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        collection = client.create_collection(name=COLLECTION_NAME)
        embedder_temp = SentenceTransformer("all-MiniLM-L6-v2")

        documents = [c["text"] for c in chunks]
        ids       = [c["chunk_id"] for c in chunks]
        metadatas = [
            {
                "product_name":           str(c.get("product_name", "")),
                "price_usd":              str(c.get("price_usd", "")),
                "rating":                 str(c.get("rating", "")),
                "sentiment_label":        str(c.get("sentiment_label", "")),
                "sentiment_score":        str(c.get("sentiment_score", "")),
                "category":               str(c.get("category", "")),
                "type":                   str(c.get("type", "")),
                "source_url":             str(c.get("source_url", "")),
            }
            for c in chunks
        ]

        embeddings = embedder_temp.encode(documents, show_progress_bar=False).tolist()

        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            collection.add(
                ids=ids[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
            )

    # Embedder
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # BM25
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25      = BM25Okapi(tokenized)

    # Gemini
    api_key = os.environ.get("GEMINI_API_KEY", "")
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", api_key)
    except Exception:
        pass

    if api_key:
        gemini_client = genai.Client(api_key=api_key)
    else:
        gemini_client = None

    return chunks, collection, embedder, bm25, gemini_client

# ─── FILTER PARSING ───────────────────────────────────────
def parse_filters(user_message):
    msg     = user_message.lower()
    filters = {}

    price_max = re.search(
        r'under\s+\$?(\d+)|less\s+than\s+\$?(\d+)|below\s+\$?(\d+)|\$?(\d+)\s+or\s+less',
        msg
    )
    if price_max:
        val = next(v for v in price_max.groups() if v is not None)
        filters["max_price"] = float(val)

    price_min = re.search(
        r'over\s+\$?(\d+)|more\s+than\s+\$?(\d+)|above\s+\$?(\d+)', msg)
    if price_min:
        val = next(v for v in price_min.groups() if v is not None)
        filters["min_price"] = float(val)

    if any(w in msg for w in ["highly rated", "top rated", "best rated", "4 star", "5 star"]):
        filters["min_rating"] = 4.0

    if any(w in msg for w in ["complaint", "problem", "issue", "bad review", "negative"]):
        filters["sentiment"] = "negative"
    elif any(w in msg for w in ["positive review", "recommended", "loved", "great review"]):
        filters["sentiment"] = "positive"

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
    query_embedding = embedder.encode([query]).tolist()

    where_clause = {}
    if "category"   in filters:
        where_clause["category"]        = {"$eq": filters["category"]}
    if "sentiment"  in filters:
        where_clause["sentiment_label"] = {"$eq": filters["sentiment"]}
    if "min_rating" in filters:
        where_clause["rating"]          = {"$gte": filters["min_rating"]}

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
    sem_scores = {doc: 1 - (dist / max_dist)
                  for doc, dist in zip(sem_docs, sem_dists)}

    bm25_scores_raw  = bm25.get_scores(query.lower().split())
    max_bm25         = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1
    bm25_norm        = [s / max_bm25 for s in bm25_scores_raw]
    top_bm25_indices = sorted(
        range(len(bm25_norm)), key=lambda i: bm25_norm[i], reverse=True
    )[:top_k * 3]

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

def build_context(results):
    parts         = []
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
                f"Sentiment: {sentiment} | Category: {category}\n"
                f"URL: {url}\n"
            )
        parts.append(f"{header}[{chunk_type.upper()}]: {doc}\n")
    return "\n---\n".join(parts)

# ─── SIDEBAR ──────────────────────────────────────────────
def render_sidebar(collection, all_chunks):
    with st.sidebar:
        st.markdown("## 🛍️ E-Commerce AI")
        st.markdown("*Powered by real Amazon data*")
        st.divider()

        # Stats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{collection.count()}</div>
                <div class="stat-label">Chunks indexed</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            products = len(set(c["product_name"] for c in all_chunks))
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{products}</div>
                <div class="stat-label">Products</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Categories
        st.markdown("### 📦 Categories")
        categories = sorted(set(c["category"] for c in all_chunks))
        for cat in categories:
            count = len([c for c in all_chunks if c["category"] == cat])
            st.markdown(f"• {cat.title()} `{count}`")

        st.divider()

        # Example queries
        st.markdown("### 💡 Try asking")
        examples = [
            "Best noise cancelling headphones for travel",
            "Wireless earbuds under $50",
            "Gaming headsets with good mic quality",
            "What are people complaining about with smartwatches?",
            "Portable chargers with fast charging",
            "Webcam recommendations for streaming",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
                st.session_state["prefill"] = ex

        st.divider()

        # Controls
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages  = []
            st.session_state.history   = []
            st.rerun()

        st.markdown(
            "<br><small>Built with Playwright · pandas · ChromaDB · "
            "sentence-transformers · BM25 · Gemini</small>",
            unsafe_allow_html=True
        )

# ─── MAIN APP ─────────────────────────────────────────────
def main():
    # Load resources once
    all_chunks, collection, embedder, bm25, gemini_model = load_all_resources()

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history  = []
    if "prefill" not in st.session_state:
        st.session_state.prefill  = ""

    # Sidebar
    render_sidebar(collection, all_chunks)

    # Header
    st.markdown("## 🛍️ E-Commerce Product Assistant")
    st.markdown(
        "Ask me anything about products — I'll search real Amazon data using "
        "**hybrid semantic + keyword search** with sentiment analysis."
    )
    st.divider()

    # API key warning
    if not gemini_model:
        st.warning(
            "⚠️ GEMINI_API_KEY not set. "
            "Run: `$env:GEMINI_API_KEY = 'your-key'` then restart."
        )
        return

    # Chat history display
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle sidebar button prefill
    prefill_value = st.session_state.pop("prefill", "")

    # Chat input
    user_input = st.chat_input(
        "Ask about any product — price, features, reviews, comparisons...",
        key="chat_input",
    ) or prefill_value

    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Parse filters and show badge
        filters = parse_filters(user_input)

        with st.chat_message("assistant"):
            # Show active filters
            if filters:
                filter_parts = []
                if "max_price"  in filters: filter_parts.append(f"💰 Under ${filters['max_price']:.0f}")
                if "min_price"  in filters: filter_parts.append(f"💰 Over ${filters['min_price']:.0f}")
                if "category"   in filters: filter_parts.append(f"📦 {filters['category'].title()}")
                if "sentiment"  in filters: filter_parts.append(f"💬 {filters['sentiment'].title()} reviews")
                if "min_rating" in filters: filter_parts.append(f"⭐ {filters['min_rating']}+ stars")
                st.markdown(
                    f'<div class="filter-box">🔍 Filters active: {" · ".join(filter_parts)}</div>',
                    unsafe_allow_html=True
                )

            # Animated thinking dots placeholder
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("""
            <div style="display:flex; align-items:center; gap:4px; padding:8px 0;">
                <span style="font-size:13px; color:var(--text-color); opacity:0.6;">Thinking</span>
                <style>
                    .dot { display:inline-block; width:6px; height:6px; border-radius:50%;
                        background:currentColor; opacity:0.4;
                        animation: blink 1.2s infinite; }
                    .dot:nth-child(2){ animation-delay:0.2s; }
                    .dot:nth-child(3){ animation-delay:0.4s; }
                    @keyframes blink {
                        0%,80%,100%{ opacity:0.2; transform:scale(0.8); }
                        40%{ opacity:1; transform:scale(1.1); }
                    }
                </style>
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
            </div>
            """, unsafe_allow_html=True)

            # Retrieve
            results = hybrid_search(
                user_input, collection, embedder,
                bm25, all_chunks, filters
            )
            context = build_context(results) if results else "No relevant products found."

            # Build prompt with recent history only
            history_text = ""
            for msg in st.session_state.history[-4:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n\n"

            full_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"{history_text}"
                f"Product data:\n\n{context}\n\n"
                f"---\nUser: {user_input}\nAssistant:"
            )

            try:
                response        = gemini_model.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=full_prompt,
                )
                assistant_reply = response.text or "I couldn't generate a response."
            except Exception as e:
                assistant_reply = f"Sorry, I encountered an error: {e}"

            # Clear thinking dots and show response
            thinking_placeholder.empty()
            st.markdown(assistant_reply)

        # Update session state
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_reply}
        )
        st.session_state.history.append({"role": "user",      "content": user_input})
        st.session_state.history.append({"role": "assistant", "content": assistant_reply})

if __name__ == "__main__":
    main()