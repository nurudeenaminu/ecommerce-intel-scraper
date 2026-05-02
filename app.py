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

SYSTEM_PROMPT = """You are a knowledgeable e-commerce product assistant.
You help users find the right products based on their needs, budget, and preferences.
You have access to real Amazon product data including descriptions, customer reviews,
prices, ratings, and sentiment analysis scores.

Guidelines:
- Answer based ONLY on the product data provided in the context
- If the context does not contain enough information, say so clearly and honestly
- Never invent product names, prices, or features not present in the context
- When recommending products, mention the price, rating, and why it fits the user's need
- Keep answers conversational, warm, and concise
- If asked something outside your dataset say: "I don't have reliable data on that in my current dataset" """

# ─── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="ShopMind AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── FULL CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif;
    background: #0f0f13;
    color: #e8e8f0;
    height: 100%;
}

[data-testid="stAppViewContainer"] {
    background: #0f0f13;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* ── Layout ── */
[data-testid="stMain"] {
    padding: 0 !important;
    background: #0f0f13;
}

[data-testid="block-container"] {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #16161e !important;
    border-right: 1px solid #2a2a35 !important;
    min-width: 280px !important;
    max-width: 280px !important;
}

[data-testid="stSidebar"] > div {
    padding: 1.5rem 1.25rem !important;
}

/* ── Chat container ── */
.chat-wrapper {
    display: flex;
    flex-direction: column;
    height: 100vh;
    max-width: 800px;
    margin: 0 auto;
    padding: 0 1rem;
}

/* ── Header ── */
.chat-header {
    padding: 1.25rem 0 1rem;
    border-bottom: 1px solid #2a2a35;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-shrink: 0;
}

.chat-header-logo {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #6c63ff, #a855f7);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}

.chat-header-text h1 {
    font-size: 17px;
    font-weight: 600;
    color: #f0f0f8;
    letter-spacing: -0.02em;
}

.chat-header-text p {
    font-size: 12px;
    color: #6b6b80;
    margin-top: 1px;
}

.online-dot {
    width: 7px; height: 7px;
    background: #22c55e;
    border-radius: 50%;
    display: inline-block;
    margin-right: 5px;
    animation: pulse-dot 2s infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Messages area ── */
.messages-area {
    flex: 1;
    overflow-y: auto;
    padding: 1.5rem 0;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    scrollbar-width: thin;
    scrollbar-color: #2a2a35 transparent;
}

/* ── Individual message rows ── */
.msg-row {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    animation: msg-in 0.25s ease-out;
}

@keyframes msg-in {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

.msg-row.user  { flex-direction: row-reverse; }
.msg-row.assistant { flex-direction: row; }

/* ── Avatars ── */
.avatar {
    width: 30px; height: 30px;
    border-radius: 50%;
    flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    font-weight: 600;
}

.avatar.user-av {
    background: linear-gradient(135deg, #6c63ff, #a855f7);
    color: white;
}

.avatar.bot-av {
    background: #1e1e2a;
    border: 1px solid #2a2a35;
    font-size: 16px;
}

/* ── Bubbles ── */
.bubble {
    max-width: 72%;
    padding: 11px 15px;
    border-radius: 18px;
    font-size: 14px;
    line-height: 1.55;
    position: relative;
}

.bubble.user-bubble {
    background: linear-gradient(135deg, #6c63ff, #7c3aed);
    color: #fff;
    border-bottom-right-radius: 4px;
}

.bubble.bot-bubble {
    background: #1c1c27;
    border: 1px solid #2a2a35;
    color: #d8d8e8;
    border-bottom-left-radius: 4px;
}

.bubble.bot-bubble strong { color: #a78bfa; }
.bubble.bot-bubble a { color: #60a5fa; text-decoration: none; }

/* ── Timestamp ── */
.msg-time {
    font-size: 10px;
    color: #44445a;
    text-align: center;
    margin: 0.25rem 0;
    font-family: 'DM Mono', monospace;
}

/* ── Filter badge ── */
.filter-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #1a2535;
    border: 1px solid #2a4060;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    color: #60a5fa;
    margin-bottom: 6px;
    font-family: 'DM Mono', monospace;
}

/* ── Thinking animation ── */
.thinking-row {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    animation: msg-in 0.2s ease-out;
}

.thinking-bubble {
    background: #1c1c27;
    border: 1px solid #2a2a35;
    border-radius: 18px;
    border-bottom-left-radius: 4px;
    padding: 13px 18px;
    display: flex;
    align-items: center;
    gap: 5px;
}

.tdot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #6c63ff;
    animation: tdot-bounce 1.3s infinite ease-in-out;
}
.tdot:nth-child(2) { animation-delay: 0.16s; }
.tdot:nth-child(3) { animation-delay: 0.32s; }

@keyframes tdot-bounce {
    0%, 60%, 100% { transform: translateY(0);   opacity: 0.4; }
    30%            { transform: translateY(-6px); opacity: 1;   }
}

/* ── Welcome state ── */
.welcome-wrap {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    padding: 2rem 0;
}

.welcome-icon {
    width: 64px; height: 64px;
    background: linear-gradient(135deg, #6c63ff22, #a855f722);
    border: 1px solid #6c63ff44;
    border-radius: 20px;
    display: flex; align-items: center; justify-content: center;
    font-size: 30px;
}

.welcome-title {
    font-size: 22px;
    font-weight: 600;
    color: #f0f0f8;
    letter-spacing: -0.03em;
}

.welcome-sub {
    font-size: 13px;
    color: #6b6b80;
    text-align: center;
    max-width: 340px;
    line-height: 1.6;
}

/* ── Quick chips ── */
.chips-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 0.5rem;
}

.chip {
    background: #1c1c27;
    border: 1px solid #2a2a35;
    border-radius: 20px;
    padding: 7px 14px;
    font-size: 12px;
    color: #a0a0b8;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
}

.chip:hover {
    border-color: #6c63ff;
    color: #c4b5fd;
    background: #1a1a2e;
}

/* ── Input area ── */
.input-area {
    padding: 1rem 0 1.25rem;
    border-top: 1px solid #2a2a35;
    flex-shrink: 0;
}

/* ── Streamlit chat input override ── */
[data-testid="stChatInput"] {
    background: #1c1c27 !important;
    border: 1px solid #2a2a35 !important;
    border-radius: 14px !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #6c63ff !important;
    box-shadow: 0 0 0 3px #6c63ff18 !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #44445a !important;
}

[data-testid="stChatInputSubmitButton"] svg {
    fill: #6c63ff !important;
}

/* ── Sidebar labels ── */
.sidebar-section {
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #44445a;
    margin: 1.25rem 0 0.6rem;
    font-family: 'DM Mono', monospace;
}

.cat-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 5px 8px;
    border-radius: 8px;
    margin-bottom: 2px;
    cursor: pointer;
    transition: background 0.12s;
    font-size: 13px;
    color: #9090a8;
}

.cat-row:hover { background: #1e1e2a; color: #d0d0e8; }

.cat-count {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #44445a;
    background: #1c1c27;
    padding: 1px 6px;
    border-radius: 10px;
}

.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 0.5rem;
}

.stat-card {
    background: #1c1c27;
    border: 1px solid #2a2a35;
    border-radius: 10px;
    padding: 10px;
    text-align: center;
}

.stat-num {
    font-size: 20px;
    font-weight: 600;
    color: #a78bfa;
    font-family: 'DM Mono', monospace;
}

.stat-lbl {
    font-size: 10px;
    color: #44445a;
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Mobile ── */
@media (max-width: 768px) {
    .chat-wrapper { padding: 0 0.75rem; }
    .bubble { max-width: 85%; font-size: 13.5px; }
    [data-testid="stSidebar"] { min-width: 100vw !important; max-width: 100vw !important; }
    .welcome-title { font-size: 18px; }
    .chips-wrap { gap: 6px; }
    .chip { font-size: 11px; padding: 6px 11px; }
}

/* ── Scrollbar ── */
.messages-area::-webkit-scrollbar { width: 4px; }
.messages-area::-webkit-scrollbar-track { background: transparent; }
.messages-area::-webkit-scrollbar-thumb { background: #2a2a35; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─── CACHED RESOURCES ─────────────────────────────────────
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
            "product_name":    str(c.get("product_name","")),
            "price_usd":       str(c.get("price_usd","")),
            "rating":          str(c.get("rating","")),
            "sentiment_label": str(c.get("sentiment_label","")),
            "sentiment_score": str(c.get("sentiment_score","")),
            "category":        str(c.get("category","")),
            "type":            str(c.get("type","")),
            "source_url":      str(c.get("source_url","")),
        } for c in chunks]
        embeddings = embedder_tmp.encode(documents, show_progress_bar=False).tolist()
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            collection.add(
                ids=ids[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
            )

    embedder  = SentenceTransformer("all-MiniLM-L6-v2")
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25      = BM25Okapi(tokenized)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", api_key)
    except Exception:
        pass

    gemini_client = genai.Client(api_key=api_key) if api_key else None
    return chunks, collection, embedder, bm25, gemini_client

# ─── FILTER PARSING ───────────────────────────────────────
def parse_filters(msg):
    m       = msg.lower()
    filters = {}
    pm = re.search(r'under\s+\$?(\d+)|less\s+than\s+\$?(\d+)|below\s+\$?(\d+)|\$?(\d+)\s+or\s+less', m)
    if pm:
        filters["max_price"] = float(next(v for v in pm.groups() if v))
    pm2 = re.search(r'over\s+\$?(\d+)|more\s+than\s+\$?(\d+)|above\s+\$?(\d+)', m)
    if pm2:
        filters["min_price"] = float(next(v for v in pm2.groups() if v))
    if any(w in m for w in ["highly rated","top rated","best rated","4 star","5 star"]):
        filters["min_rating"] = 4.0
    if any(w in m for w in ["complaint","problem","issue","bad review","negative"]):
        filters["sentiment"] = "negative"
    elif any(w in m for w in ["positive review","recommended","loved","great review"]):
        filters["sentiment"] = "positive"
    cats = {
        "headphone":"wireless headphones","speaker":"bluetooth speakers",
        "earbud":"wireless earbuds","noise cancel":"noise cancelling headphones",
        "gaming":"gaming headsets","watch":"smart watches",
        "keyboard":"wireless keyboards","webcam":"webcams for streaming",
        "charger":"portable chargers","laptop stand":"laptop stands",
    }
    for kw, cat in cats.items():
        if kw in m:
            filters["category"] = cat
            break
    return filters

# ─── HYBRID SEARCH ────────────────────────────────────────
def hybrid_search(query, collection, embedder, bm25, all_chunks, filters):
    qe = embedder.encode([query]).tolist()
    wc = {}
    if "category"  in filters: wc["category"]        = {"$eq": filters["category"]}
    if "sentiment" in filters: wc["sentiment_label"]  = {"$eq": filters["sentiment"]}
    kw = {"query_embeddings": qe,
          "n_results": min(TOP_K*3, collection.count()),
          "include": ["documents","metadatas","distances"]}
    if wc: kw["where"] = wc
    try:
        sr = collection.query(**kw)
    except Exception:
        sr = collection.query(query_embeddings=qe,
                              n_results=min(TOP_K*3,collection.count()),
                              include=["documents","metadatas","distances"])
    docs, metas, dists = sr["documents"][0], sr["metadatas"][0], sr["distances"][0]
    md = max(dists) if dists else 1
    ss = {d: 1-(dist/md) for d, dist in zip(docs, dists)}

    br  = bm25.get_scores(query.lower().split())
    mb  = max(br) if max(br) > 0 else 1
    bn  = [s/mb for s in br]
    tbi = sorted(range(len(bn)), key=lambda i: bn[i], reverse=True)[:TOP_K*3]

    combined = {}
    for doc, meta, score in zip(docs, metas, ss.values()):
        combined[doc] = {"meta": meta, "score": (1-BM25_WEIGHT)*score}
    for idx in tbi:
        doc  = all_chunks[idx]["text"]
        meta = {k: str(v) for k,v in all_chunks[idx].items() if k != "text"}
        if doc in combined:
            combined[doc]["score"] += BM25_WEIGHT * bn[idx]
        else:
            combined[doc] = {"meta": meta, "score": BM25_WEIGHT * bn[idx]}

    filtered = {}
    for doc, data in combined.items():
        price = float(data["meta"].get("price_usd", 0) or 0)
        if "max_price" in filters and price > 0 and price > filters["max_price"]: continue
        if "min_price" in filters and price > 0 and price < filters["min_price"]: continue
        filtered[doc] = data
    if not filtered: filtered = combined

    return sorted(filtered.items(), key=lambda x: x[1]["score"], reverse=True)[:TOP_K]

def build_context(results):
    parts, seen = [], set()
    for doc, data in results:
        meta  = data["meta"]
        name  = meta.get("product_name","Unknown")
        price = meta.get("price_usd","0")
        hdr   = ""
        if name not in seen:
            seen.add(name)
            try:    ps = f"USD {float(price):.2f}" if float(price) > 0 else "See site"
            except: ps = "See site"
            hdr = (f"Product: {name}\nPrice: {ps} | Rating: {meta.get('rating','N/A')}/5 | "
                   f"Sentiment: {meta.get('sentiment_label','N/A')}\nURL: {meta.get('source_url','')}\n")
        parts.append(f"{hdr}[{meta.get('type','').upper()}]: {doc}\n")
    return "\n---\n".join(parts)

# ─── SIDEBAR ──────────────────────────────────────────────
def render_sidebar(collection, all_chunks):
    with st.sidebar:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;">
            <div style="width:36px;height:36px;background:linear-gradient(135deg,#6c63ff,#a855f7);
                        border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px;">🛍️</div>
            <div>
                <div style="font-size:15px;font-weight:600;color:#f0f0f8;">ShopMind AI</div>
                <div style="font-size:11px;color:#44445a;">Powered by real Amazon data</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        products = len(set(c.get("product_name","") for c in all_chunks))
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-num">{collection.count()}</div>
                <div class="stat-lbl">Chunks</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">{products}</div>
                <div class="stat-lbl">Products</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Categories</div>', unsafe_allow_html=True)
        categories = sorted(set(c.get("category","") for c in all_chunks if c.get("category")))
        for cat in categories:
            count = len([c for c in all_chunks if c.get("category") == cat])
            label = cat.title()
            if st.button(f"{label}  {count}", key=f"cat_{cat}", use_container_width=True):
                st.session_state.prefill = f"What are the best {cat}?"

        st.markdown('<div class="sidebar-section">Actions</div>', unsafe_allow_html=True)
        if st.button("🗑️  Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history  = []
            st.rerun()

        st.markdown("""
        <div style="margin-top:2rem;padding-top:1rem;border-top:1px solid #2a2a35;">
            <div style="font-size:10px;color:#2a2a35;font-family:'DM Mono',monospace;line-height:1.8;">
                scraper · cleaner · chunker<br>
                chromadb · bm25 · gemini 2.5
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── RENDER MESSAGES ──────────────────────────────────────
def render_messages(messages):
    for i, msg in enumerate(messages):
        role    = msg["role"]
        content = msg["content"]
        ts      = msg.get("time", "")

        if role == "user":
            st.markdown(f"""
            <div class="msg-row user">
                <div class="avatar user-av">U</div>
                <div class="bubble user-bubble">{content}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Show filter badge if present
            badge_html = ""
            if msg.get("filters"):
                parts = []
                f = msg["filters"]
                if "max_price"  in f: parts.append(f"💰 Under ${f['max_price']:.0f}")
                if "min_price"  in f: parts.append(f"💰 Over ${f['min_price']:.0f}")
                if "category"   in f: parts.append(f"📦 {f['category'].title()}")
                if "sentiment"  in f: parts.append(f"💬 {f['sentiment'].title()}")
                if "min_rating" in f: parts.append(f"⭐ {f['min_rating']}+")
                if parts:
                    badge_html = f'<div class="filter-badge">🔍 {" · ".join(parts)}</div>'

            st.markdown(f"""
            <div class="msg-row assistant">
                <div class="avatar bot-av">🛍️</div>
                <div style="display:flex;flex-direction:column;gap:4px;">
                    {badge_html}
                    <div class="bubble bot-bubble">{content}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Timestamp every few messages
        if ts and (i == len(messages)-1 or i % 6 == 0):
            st.markdown(f'<div class="msg-time">{ts}</div>', unsafe_allow_html=True)

# ─── QUICK CHIPS ──────────────────────────────────────────
QUICK_CHIPS = [
    "🎧 Best noise cancelling headphones",
    "🎮 Gaming headsets under $100",
    "📱 Wireless earbuds for workouts",
    "⌚ Highly rated smartwatches",
    "🔋 Fast charging portable chargers",
    "📷 Webcams for streaming",
]

def render_welcome():
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-icon">🛍️</div>
        <div class="welcome-title">ShopMind AI</div>
        <div class="welcome-sub">Ask me anything about electronics and gadgets.
        I search real Amazon data to find what fits your needs and budget.</div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    for i, chip in enumerate(QUICK_CHIPS):
        with cols[i % 3]:
            if st.button(chip, key=f"chip_{i}", use_container_width=True):
                st.session_state.prefill = chip

# ─── MAIN ─────────────────────────────────────────────────
def main():
    all_chunks, collection, embedder, bm25, gemini_client = load_all_resources()

    if "messages" not in st.session_state: st.session_state.messages = []
    if "history"  not in st.session_state: st.session_state.history  = []
    if "prefill"  not in st.session_state: st.session_state.prefill  = ""

    render_sidebar(collection, all_chunks)

    if not gemini_client:
        st.error("⚠️ GEMINI_API_KEY not configured.")
        return

    # ── Main chat layout ──
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="chat-header">
        <div class="chat-header-logo">🛍️</div>
        <div class="chat-header-text">
            <h1>ShopMind AI</h1>
            <p><span class="online-dot"></span>Online · Real Amazon data · Hybrid search</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Messages area
    st.markdown('<div class="messages-area" id="msgs">', unsafe_allow_html=True)

    if not st.session_state.messages:
        render_welcome()
    else:
        render_messages(st.session_state.messages)

    st.markdown('</div>', unsafe_allow_html=True)

    # Input
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    prefill_val = st.session_state.pop("prefill", "")
    user_input  = st.chat_input(
        "Ask about any product — price, features, reviews...",
        key="chat_input"
    ) or prefill_val
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Process message ──
    if user_input:
        ts      = time.strftime("%I:%M %p")
        filters = parse_filters(user_input)

        st.session_state.messages.append({
            "role": "user", "content": user_input, "time": ts
        })

        # Show thinking dots
        thinking = st.empty()
        thinking.markdown("""
        <div class="thinking-row">
            <div class="avatar bot-av">🛍️</div>
            <div class="thinking-bubble">
                <div class="tdot"></div>
                <div class="tdot"></div>
                <div class="tdot"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Retrieve + generate
        results = hybrid_search(user_input, collection, embedder, bm25, all_chunks, filters)
        context = build_context(results) if results else "No relevant products found."

        history_text = ""
        for msg in st.session_state.history[-4:]:
            role          = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n\n"

        full_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"{history_text}"
            f"Product data:\n\n{context}\n\n"
            f"---\nUser: {user_input}\nAssistant:"
        )

        try:
            response        = gemini_client.models.generate_content(
                model=GEMINI_MODEL, contents=full_prompt)
            assistant_reply = response.text or "I couldn't generate a response."
        except Exception as e:
            assistant_reply = f"Sorry, I hit an error: {e}"

        thinking.empty()

        st.session_state.messages.append({
            "role":    "assistant",
            "content": assistant_reply,
            "filters": filters if filters else None,
            "time":    time.strftime("%I:%M %p"),
        })
        st.session_state.history.append({"role": "user",      "content": user_input})
        st.session_state.history.append({"role": "assistant", "content": assistant_reply})

        st.rerun()

if __name__ == "__main__":
    main()