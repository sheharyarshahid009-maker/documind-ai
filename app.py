import streamlit as st
import pdfplumber
import chromadb
from groq import Groq
import tempfile

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="DocuMind AI — RAG Document Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ PREMIUM CSS ============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

    * { font-family: 'Space Grotesk', sans-serif; }

    .stApp {
        background: radial-gradient(ellipse at top, #1a1a2e 0%, #0a0a0f 60%, #000000 100%);
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffd700 0%, #ffaa00 40%, #ff6b00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-top: 8px;
        margin-bottom: 30px;
    }

    .feature-card {
        background: rgba(255, 215, 0, 0.04);
        border: 1px solid rgba(255, 215, 0, 0.15);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }

    .feature-card:hover {
        transform: translateY(-3px);
        border-color: rgba(255, 215, 0, 0.4);
    }

    .feature-card .icon { font-size: 2rem; margin-bottom: 8px; }
    .feature-card .title { color: #ffd700; font-weight: 600; font-size: 0.95rem; }
    .feature-card .desc { color: #888; font-size: 0.8rem; margin-top: 4px; }

    .stat-box {
        background: linear-gradient(135deg, rgba(255,215,0,0.08), rgba(255,107,0,0.04));
        border: 1px solid rgba(255, 215, 0, 0.2);
        border-radius: 12px;
        padding: 12px 16px;
        text-align: center;
    }
    .stat-box .num { font-size: 1.6rem; font-weight: 700; color: #ffd700; }
    .stat-box .lbl { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }

    .stChatMessage {
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.06);
    }

    .source-chip {
        display: inline-block;
        background: rgba(255, 215, 0, 0.1);
        border: 1px solid rgba(255, 215, 0, 0.3);
        color: #ffd700;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        margin: 2px 4px 2px 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #ffd700, #ff8c00);
        color: #000;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(255, 215, 0, 0.3);
        color: #000;
    }

    .footer {
        text-align: center;
        color: #555;
        font-size: 0.85rem;
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    .footer b { color: #ffd700; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ============ CONSTANTS ============
FALLBACK_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]

# ============ CHROMADB ============
@st.cache_resource
def get_collection():
    tmpdir = tempfile.mkdtemp()
    client = chromadb.PersistentClient(path=tmpdir)
    return client.get_or_create_collection(name="documind_docs")

# ============ HELPERS ============
def extract_pages(pdf_file):
    """PDF se har page ka text nikalo — 2000 pages bhi handle karega."""
    pages = []
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i + 1, text))
    return pages

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks

def process_documents(files, collection, chunk_size, overlap):
    """Documents ko chunks me tod ke ChromaDB me save karo (batch me)."""
    # Purana data clear
    try:
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    all_docs, all_metas, all_ids = [], [], []
    total_pages = 0

    for f in files:
        pages = extract_pages(f)
        total_pages += len(pages)
        for page_num, page_text in pages:
            for c_idx, chunk in enumerate(chunk_text(page_text, chunk_size, overlap)):
                all_docs.append(chunk)
                all_metas.append({"filename": f.name, "page": page_num})
                all_ids.append(f"{f.name}_p{page_num}_c{c_idx}")

    # Batch me add karo — bari PDFs ke liye safe
    bar = st.progress(0, text="📚 Documents ko index kiya ja raha hai...")
    batch = 100
    for i in range(0, len(all_docs), batch):
        j = min(i + batch, len(all_docs))
        collection.add(documents=all_docs[i:j], metadatas=all_metas[i:j], ids=all_ids[i:j])
        bar.progress(j / len(all_docs), text=f"📚 Indexing... {j}/{len(all_docs)} chunks")
    bar.empty()
    return len(all_docs), total_pages

def ask_groq(api_key, prompt):
    client = Groq(api_key=api_key)
    last_err = None
    for model in FALLBACK_MODELS:
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1500
            )
            return r.choices[0].message.content, model
        except Exception as e:
            last_err = e
    raise last_err

# ============ SESSION STATE ============
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "stats" not in st.session_state:
    st.session_state.stats = {"files": 0, "pages": 0, "chunks": 0}

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("### 🔑 Setup")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
            st.success("✅ API key loaded")
        except Exception:
            st.info("👆 API key dalo (console.groq.com se free)")

    st.markdown("---")
    st.markdown("### 📄 Documents")
    uploaded = st.file_uploader("PDF upload karo", type=["pdf"], accept_multiple_files=True)

    st.markdown("### ⚙️ Settings")
    chunk_size = st.slider("Chunk size", 500, 2000, 1000, 100)
    top_k = st.slider("Sources per answer", 1, 10, 5)

    if st.button("🚀 Process Documents", use_container_width=True):
        if not uploaded:
            st.warning("Pehle PDF upload karo!")
        else:
            with st.spinner("Processing..."):
                n_chunks, n_pages = process_documents(uploaded, get_collection(), chunk_size, 200)
            st.session_state.indexed = True
            st.session_state.stats = {"files": len(uploaded), "pages": n_pages, "chunks": n_chunks}
            st.session_state.messages = []
            st.success(f"✅ {len(uploaded)} files, {n_pages} pages, {n_chunks} chunks indexed!")

    if st.session_state.indexed:
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="stat-box"><div class="num">{st.session_state.stats["files"]}</div><div class="lbl">Files</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-box"><div class="num">{st.session_state.stats["pages"]}</div><div class="lbl">Pages</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-box"><div class="num">{st.session_state.stats["chunks"]}</div><div class="lbl">Chunks</div></div>', unsafe_allow_html=True)

        if st.button("🗑️ Clear All", use_container_width=True):
            try:
                ids = get_collection().get()["ids"]
                if ids:
                    get_collection().delete(ids=ids)
            except Exception:
                pass
            st.session_state.indexed = False
            st.session_state.messages = []
            st.rerun()

# ============ HERO ============
st.markdown('<div class="hero-title">🧠 DocuMind AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">RAG-powered Document Q&A &nbsp;•&nbsp; 2000 pages? No problem. 🚀</div>', unsafe_allow_html=True)

# ============ WELCOME CARDS ============
if not st.session_state.messages and not st.session_state.indexed:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="feature-card"><div class="icon">📄</div><div class="title">Upload PDFs</div><div class="desc">Kitni bhi bari files — reports, books, manuals</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-card"><div class="icon">🔍</div><div class="title">Ask Anything</div><div class="desc">AI sirf tumhare documents se jawab dega</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="feature-card"><div class="icon">📎</div><div class="title">Cited Sources</div><div class="desc">Har jawab ke saath file + page number</div></div>', unsafe_allow_html=True)
    st.info("👈 Sidebar se PDF upload karo aur **Process Documents** dabao!")

# ============ CHAT HISTORY ============
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            chips = "".join([f'<span class="source-chip">📄 {s["filename"]} — p.{s["page"]}</span>' for s in msg["sources"]])
            st.markdown(chips, unsafe_allow_html=True)

# ============ CHAT INPUT ============
if prompt := st.chat_input("Apne documents se sawal poocho..."):
    if not api_key:
        st.warning("⚠️ Pehle Groq API key dalo (sidebar me)")
        st.stop()
    if not st.session_state.indexed:
        st.warning("⚠️ Pehle documents process karo (sidebar me)")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Documents me dhoond raha hun..."):
            results = get_collection().query(query_texts=[prompt], n_results=top_k)
            docs = results["documents"][0]
            metas = results["metadatas"][0]

            if not docs:
                answer = "Is ka jawab documents me nahi mila. Kuch aur poocho!"
                sources = []
            else:
                context = "\n\n---\n\n".join(
                    [f"[Source: {m['filename']}, Page {m['page']}]\n{d}" for d, m in zip(docs, metas)]
                )
                rag_prompt = f"""You are DocuMind AI, a precise document Q&A assistant.

Answer the question using ONLY the context below. If the answer is not in the context, say "Is ka jawab documents me nahi mila."
Be concise but complete. Mention page numbers when relevant.

CONTEXT:
{context}

QUESTION: {prompt}

ANSWER:"""
                try:
                    answer, model_used = ask_groq(api_key, rag_prompt)
                except Exception as e:
                    answer = f"❌ API error: {str(e)[:200]}"
                    model_used = None

                seen = set()
                sources = []
                for m in metas:
                    key = (m["filename"], m["page"])
                    if key not in seen:
                        seen.add(key)
                        sources.append({"filename": m["filename"], "page": m["page"]})

            st.markdown(answer)
            if sources:
                st.markdown("**📎 Sources:**")
                chips = "".join([f'<span class="source-chip">📄 {s["filename"]} — p.{s["page"]}</span>' for s in sources])
                st.markdown(chips, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

# ============ FOOTER ============
st.markdown('<div class="footer">Crafted with ❤️ by <b>Engr. Muhammad Sheharyar Khan</b> &nbsp;•&nbsp; Powered by RAG + Groq AI</div>', unsafe_allow_html=True)
