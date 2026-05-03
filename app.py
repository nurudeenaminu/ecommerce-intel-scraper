import json
import os
import re
import time
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from google import genai

# ─── CONFIG ───────────────────────────────────────────────
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "amazon_products"
JSONL_FILE      = "data/chunks/products_chunks.jsonl"
TOP_K           = 5
BM25_WEIGHT     = 0.3
GEMINI_MODEL    = "gemini-2.5-flash"

def build_system_prompt(chunk_count, product_count, categories):
    cat_list = "\n".join(f"  - {cat.title()}" for cat in sorted(categories))
    return f"""You are a knowledgeable e-commerce product assistant for ShopMind AI.

DATASET FACTS — answer these precisely when asked:
- Total chunks indexed: {chunk_count}
- Total unique products: {product_count}
- Total categories: {len(categories)}
- Categories available:
{cat_list}

Guidelines:
- Answer based ONLY on the product data provided in the context
- When asked how many products, chunks, or categories you have, use the DATASET FACTS above
- If context lacks info for a specific product question, say so clearly
- Never invent product names, prices, or features
- Mention price, rating, and fit when recommending
- Be conversational, warm, and concise
- Outside dataset: say you don't have reliable data on that"""

st.set_page_config(
    page_title="ShopMind AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────
st.markdown(f"<style>{open('style.css').read()}</style>", unsafe_allow_html=True)

# ─── RESOURCE LOADER ──────────────────────────────────────
@st.cache_resource
def load_all_resources():
    chunks = []
    with open(JSONL_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() == 0:
            raise ValueError("empty")
    except Exception:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection   = client.create_collection(name=COLLECTION_NAME)
        embedder_tmp = SentenceTransformer("all-MiniLM-L6-v2")
        documents    = [c["text"]     for c in chunks]
        ids          = [c["chunk_id"] for c in chunks]
        metadatas    = [{
            "product_name":    str(c.get("product_name", "")),
            "price_usd":       str(c.get("price_usd", "")),
            "rating":          str(c.get("rating", "")),
            "sentiment_label": str(c.get("sentiment_label", "")),
            "sentiment_score": str(c.get("sentiment_score", "")),
            "category":        str(c.get("category", "")),
            "type":            str(c.get("type", "")),
            "source_url":      str(c.get("source_url", "")),
        } for c in chunks]
        embeddings = embedder_tmp.encode(documents, show_progress_bar=False).tolist()
        for i in range(0, len(chunks), 50):
            collection.add(
                ids=ids[i:i+50],
                documents=documents[i:i+50],
                metadatas=metadatas[i:i+50],
                embeddings=embeddings[i:i+50],
            )

    embedder  = SentenceTransformer("all-MiniLM-L6-v2")
    bm25      = BM25Okapi([c["text"].lower().split() for c in chunks])

    api_key = os.environ.get("GEMINI_API_KEY", "")
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", api_key)
    except Exception:
        pass

    return chunks, collection, embedder, bm25, genai.Client(api_key=api_key) if api_key else None

# ─── HELPERS ──────────────────────────────────────────────
def parse_filters(msg):
    m, filters = msg.lower(), {}
    pm = re.search(r'under\s+\$?(\d+)|less\s+than\s+\$?(\d+)|below\s+\$?(\d+)|\$?(\d+)\s+or\s+less', m)
    if pm: filters["max_price"] = float(next(v for v in pm.groups() if v))
    pm2 = re.search(r'over\s+\$?(\d+)|more\s+than\s+\$?(\d+)|above\s+\$?(\d+)', m)
    if pm2: filters["min_price"] = float(next(v for v in pm2.groups() if v))
    if any(w in m for w in ["highly rated","top rated","4 star","5 star"]): filters["min_rating"] = 4.0
    if any(w in m for w in ["complaint","problem","issue","bad review","negative"]): filters["sentiment"] = "negative"
    elif any(w in m for w in ["positive review","recommended","loved"]): filters["sentiment"] = "positive"
    cats = {"headphone":"wireless headphones","speaker":"bluetooth speakers",
            "earbud":"wireless earbuds","noise cancel":"noise cancelling headphones",
            "gaming":"gaming headsets","watch":"smart watches","keyboard":"wireless keyboards",
            "webcam":"webcams for streaming","charger":"portable chargers","laptop stand":"laptop stands"}
    for kw, cat in cats.items():
        if kw in m: filters["category"] = cat; break
    return filters

def hybrid_search(query, collection, embedder, bm25, all_chunks, filters):
    qe = embedder.encode([query]).tolist()
    wc = {}
    if "category"  in filters: wc["category"]       = {"$eq": filters["category"]}
    if "sentiment" in filters: wc["sentiment_label"] = {"$eq": filters["sentiment"]}
    kw = {"query_embeddings": qe, "n_results": min(TOP_K*3, collection.count()),
          "include": ["documents","metadatas","distances"]}
    if wc: kw["where"] = wc
    try:    sr = collection.query(**kw)
    except: sr = collection.query(query_embeddings=qe, n_results=min(TOP_K*3, collection.count()),
                                   include=["documents","metadatas","distances"])
    docs, metas, dists = sr["documents"][0], sr["metadatas"][0], sr["distances"][0]
    md = max(dists) if dists else 1
    combined = {d: {"meta": m, "score": (1-BM25_WEIGHT)*(1-dist/md)}
                for d, m, dist in zip(docs, metas, dists)}
    br  = bm25.get_scores(query.lower().split())
    mb  = max(br) if max(br) > 0 else 1
    bn  = [s/mb for s in br]
    for idx in sorted(range(len(bn)), key=lambda i: bn[i], reverse=True)[:TOP_K*3]:
        doc  = all_chunks[idx]["text"]
        meta = {k: str(v) for k,v in all_chunks[idx].items() if k != "text"}
        if doc in combined: combined[doc]["score"] += BM25_WEIGHT * bn[idx]
        else: combined[doc] = {"meta": meta, "score": BM25_WEIGHT * bn[idx]}
    filtered = {d: v for d,v in combined.items()
                if not ("max_price" in filters and float(v["meta"].get("price_usd",0) or 0) > filters["max_price"] > 0)
                and not ("min_price" in filters and float(v["meta"].get("price_usd",0) or 0) < filters["min_price"] > 0)}
    return sorted((filtered or combined).items(), key=lambda x: x[1]["score"], reverse=True)[:TOP_K]

def build_context(results):
    parts, seen = [], set()
    for doc, data in results:
        meta = data["meta"]
        name = meta.get("product_name","Unknown")
        hdr  = ""
        if name not in seen:
            seen.add(name)
            try:    ps = f"USD {float(meta.get('price_usd',0)):.2f}" if float(meta.get('price_usd',0)) > 0 else "See site"
            except: ps = "See site"
            hdr = (f"Product: {name}\nPrice: {ps} | Rating: {meta.get('rating','N/A')}/5 | "
                   f"Sentiment: {meta.get('sentiment_label','N/A')}\nURL: {meta.get('source_url','')}\n")
        parts.append(f"{hdr}[{meta.get('type','').upper()}]: {doc}\n")
    return "\n---\n".join(parts)

# ─── SIDEBAR ──────────────────────────────────────────────
def render_sidebar(collection, all_chunks):
    with st.sidebar:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.2rem;">
            <div style="width:36px;height:36px;background:linear-gradient(135deg,#6c63ff,#a855f7);
                 border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px;">🛍️</div>
            <div>
                <div style="font-size:15px;font-weight:600;color:#f0f0f8;">ShopMind AI</div>
                <div style="font-size:11px;color:#44445a;">Real Amazon data</div>
            </div>
        </div>""", unsafe_allow_html=True)

        products = len(set(c.get("product_name","") for c in all_chunks))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="scard"><div class="snum">{collection.count()}</div>'
                        f'<div class="slbl">Chunks</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="scard"><div class="snum">{products}</div>'
                        f'<div class="slbl">Products</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="slabel">📦 Categories</div>', unsafe_allow_html=True)
        for cat in sorted(set(c.get("category","") for c in all_chunks if c.get("category"))):
            count = sum(1 for c in all_chunks if c.get("category") == cat)
            if st.button(f"{cat.title()}  ·  {count}", key=f"c_{cat}", use_container_width=True):
                st.session_state.prefill = f"What are the best {cat}?"

        st.markdown('<div class="slabel">💡 Try asking</div>', unsafe_allow_html=True)
        for q in ["Best noise cancelling for travel",
                  "Wireless earbuds under $50",
                  "Gaming headsets with good mic",
                  "Smartwatch complaints",
                  "Fast charging portable chargers",
                  "Webcams for streaming"]:
            if st.button(q, key=f"q_{q[:15]}", use_container_width=True):
                st.session_state.prefill = q

        st.markdown('<div class="slabel">⚙️ Actions</div>', unsafe_allow_html=True)
        if st.button("🗑️  Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history  = []
            st.rerun()

        st.markdown("""<div style="margin-top:2rem;font-size:10px;color:#2a2a35;line-height:1.9;">
            Playwright · pandas · TextBlob<br>
            ChromaDB · BM25 · Gemini 2.5<br>
            sentence-transformers · tiktoken
        </div>""", unsafe_allow_html=True)

# ─── MAIN ─────────────────────────────────────────────────
def main():
    all_chunks, collection, embedder, bm25, gemini_client = load_all_resources()

    if "messages" not in st.session_state: st.session_state.messages = []
    if "history"  not in st.session_state: st.session_state.history  = []
    if "prefill"  not in st.session_state: st.session_state.prefill  = ""

    render_sidebar(collection, all_chunks)

    if not gemini_client:
        st.error("GEMINI_API_KEY not set.")
        return

    # Header
    chunk_count   = collection.count()
    product_count = len(set(c.get("product_name","") for c in all_chunks))
    cat_count     = len(set(c.get("category","") for c in all_chunks if c.get("category")))

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;padding-bottom:1rem;
        border-bottom:1px solid #2a2a35;margin-bottom:1.5rem;">
        <div style="width:40px;height:40px;background:linear-gradient(135deg,#6c63ff,#a855f7);
            border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;">🛍️</div>
        <div>
            <div style="font-size:18px;font-weight:600;color:#f0f0f8;letter-spacing:-0.02em;">ShopMind AI</div>
            <div style="font-size:12px;color:#6b6b80;">
                <span style="display:inline-block;width:7px;height:7px;background:#22c55e;
                border-radius:50%;margin-right:5px;"></span>
                Online · {chunk_count} chunks · {product_count} products · {cat_count} categories
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Welcome screen
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome">
            <h2>What can I help you find?</h2>
            Ask about any product — price, features, reviews, or comparisons.<br>
            I search real Amazon data using semantic + keyword hybrid search.
        </div>""", unsafe_allow_html=True)

    # Render chat history
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role, avatar="👤" if role=="user" else "🛍️"):
            if role == "assistant" and msg.get("filters"):
                f      = msg["filters"]
                parts  = []
                if "max_price"  in f: parts.append(f"💰 Under ${f['max_price']:.0f}")
                if "min_price"  in f: parts.append(f"💰 Over ${f['min_price']:.0f}")
                if "category"   in f: parts.append(f"📦 {f['category'].title()}")
                if "sentiment"  in f: parts.append(f"💬 {f['sentiment'].title()}")
                if "min_rating" in f: parts.append(f"⭐ {f['min_rating']}+")
                if parts:
                    st.markdown(f'<div class="fbadge">🔍 {" · ".join(parts)}</div>',
                                unsafe_allow_html=True)
            st.markdown(msg["content"])

    # Input
    prefill    = st.session_state.pop("prefill", "")
    user_input = st.chat_input("Ask about any product...", key="chat_input") or prefill

    if user_input:
        filters = parse_filters(user_input)
        st.session_state.messages.append({"role":"user","content":user_input})
        st.session_state.history.append({"role":"user","content":user_input})

        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🛍️"):
            # Filter badge
            if filters:
                parts = []
                if "max_price"  in filters: parts.append(f"💰 Under ${filters['max_price']:.0f}")
                if "min_price"  in filters: parts.append(f"💰 Over ${filters['min_price']:.0f}")
                if "category"   in filters: parts.append(f"📦 {filters['category'].title()}")
                if "sentiment"  in filters: parts.append(f"💬 {filters['sentiment'].title()}")
                if "min_rating" in filters: parts.append(f"⭐ {filters['min_rating']}+")
                if parts:
                    st.markdown(f'<div class="fbadge">🔍 {" · ".join(parts)}</div>',
                                unsafe_allow_html=True)

            # Thinking dots
            thinking = st.empty()
            thinking.markdown("""
            <div class="tdots">
                <div class="tdot"></div>
                <div class="tdot"></div>
                <div class="tdot"></div>
            </div>""", unsafe_allow_html=True)

            # Retrieve + generate
            results = hybrid_search(user_input, collection, embedder, bm25, all_chunks, filters)
            context = build_context(results) if results else "No relevant products found."

            history_text = ""
            for m in st.session_state.history[-4:]:
                history_text += f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}\n\n"

            system_prompt = build_system_prompt(
                collection.count(),
                len(set(c.get("product_name","") for c in all_chunks)),
                set(c.get("category","") for c in all_chunks if c.get("category"))
            )
            prompt = (f"{system_prompt}\n\n{history_text}"
                      f"Product data:\n\n{context}\n\n---\nUser: {user_input}\nAssistant:")
            try:
                resp   = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                reply  = resp.text or "I could not generate a response."
            except Exception as e:
                reply  = f"Error: {e}"

            thinking.empty()
            st.markdown(reply)

        st.session_state.messages.append({
            "role":"assistant","content":reply,"filters":filters or None
        })
        st.session_state.history.append({"role":"assistant","content":reply})

if __name__ == "__main__":
    main()