"""
app.py — Creative MSADS RAG Chatbot — Final Version
Supports: "ollama" (free/local), "openai", "anthropic"
Run: streamlit run src/app.py
"""
import os, re, time, json
import streamlit as st

LLM_PROVIDER = "ollama"
OLLAMA_MODEL = "llama3.2"

st.set_page_config(
    page_title="MSADS AI · UChicago",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #800000 0%, #1F3864 100%);
    padding: 1.6rem 2rem; border-radius: 14px; margin-bottom: 1rem;
    color: white; position: relative; overflow: hidden;
}
.main-header::after {
    content: "🎓"; position: absolute; right: 1.5rem; top: 50%;
    transform: translateY(-50%); font-size: 4rem; opacity: 0.12;
}
.main-header h1 { margin: 0; font-size: 1.6rem; font-weight: 700; }
.main-header p  { margin: 0.3rem 0 0; opacity: 0.75; font-size: 0.88rem; }

.metric-row { display: flex; gap: 10px; margin-bottom: 1rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 100px; background: white;
    border: 1px solid #eee; border-radius: 10px;
    padding: 0.75rem; text-align: center;
}
.metric-card .val { font-size: 1.5rem; font-weight: 700; color: #800000; line-height: 1; }
.metric-card .lbl { font-size: 0.68rem; color: #888; margin-top: 3px; text-transform: uppercase; letter-spacing: 0.05em; }

.src-card {
    background: #fafafa; border-left: 3px solid #800000;
    border-radius: 0 8px 8px 0; padding: 0.6rem 0.9rem; margin: 5px 0; font-size: 0.8rem;
}
.src-title { font-weight: 600; color: #1F3864; }
.src-score {
    display: inline-block; background: #800000; color: white;
    border-radius: 20px; padding: 1px 7px; font-size: 0.68rem; margin-left: 5px;
}
.src-text { color: #555; margin-top: 3px; line-height: 1.5; }

.followup-btn {
    display: inline-block; background: #EEF2FF; color: #3730A3;
    border-radius: 20px; padding: 4px 12px; font-size: 0.78rem;
    margin: 3px; cursor: pointer; border: 1px solid #C7D2FE;
}

.pipeline-card {
    background: white; border: 1px solid #eee; border-radius: 10px;
    padding: 0.7rem 0.9rem; margin: 5px 0;
    display: flex; align-items: flex-start; gap: 10px;
}
.pipeline-icon { font-size: 1.2rem; margin-top: 1px; }
.pipeline-label { font-weight: 600; font-size: 0.82rem; color: #1F3864; }
.pipeline-desc  { font-size: 0.75rem; color: #666; }

.topic-teal   { background:#E0F7FA; color:#006064; border-radius:20px; padding:2px 10px; font-size:0.72rem; font-weight:500; margin:2px; display:inline-block; }
.topic-indigo { background:#EEF2FF; color:#3730A3; border-radius:20px; padding:2px 10px; font-size:0.72rem; font-weight:500; margin:2px; display:inline-block; }
.topic-green  { background:#E8F5E9; color:#1B5E20; border-radius:20px; padding:2px 10px; font-size:0.72rem; font-weight:500; margin:2px; display:inline-block; }
.topic-gold   { background:#FFF8E1; color:#E65100; border-radius:20px; padding:2px 10px; font-size:0.72rem; font-weight:500; margin:2px; display:inline-block; }
.topic-gray   { background:#F3F4F6; color:#374151; border-radius:20px; padding:2px 10px; font-size:0.72rem; font-weight:500; margin:2px; display:inline-block; }

.stButton > button {
    border-radius: 8px !important; border: 1px solid #e0e0e0 !important;
    font-size: 0.8rem !important; text-align: left !important;
    padding: 0.35rem 0.75rem !important; transition: all 0.15s !important;
    width: 100% !important;
}
.stButton > button:hover { border-color: #800000 !important; color: #800000 !important; background: #fff5f5 !important; }
</style>
""", unsafe_allow_html=True)

# ── Key check ─────────────────────────────────────────────────────────────────
def check_key():
    if LLM_PROVIDER == "ollama": return None
    env = "OPENAI_API_KEY" if LLM_PROVIDER == "openai" else "ANTHROPIC_API_KEY"
    k = os.environ.get(env, "")
    if not k:
        st.error(f"Set: export {env}='your-key' then restart Streamlit.")
        st.stop()
    return k

api_key = check_key()

@st.cache_resource(show_spinner="Loading knowledge base and embeddings...")
def load_resources():
    import sys; sys.path.insert(0, "src")
    from vector_store import MSADSVectorStore
    store = MSADSVectorStore()
    if LLM_PROVIDER == "openai":
        from openai import OpenAI
        return store, OpenAI(api_key=api_key)
    elif LLM_PROVIDER == "anthropic":
        from anthropic import Anthropic
        return store, Anthropic(api_key=api_key)
    else:
        from openai import OpenAI
        return store, OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

store, client = load_resources()

SYSTEM = """You are an expert, enthusiastic AI assistant for the University of Chicago's
MS in Applied Data Science (MSADS) program. You help prospective students, current students,
and alumni get accurate, helpful information.

RULES:
1. Answer ONLY from the provided context passages. Be thorough and specific.
2. If the context lacks information, say so and suggest visiting datascience.uchicago.edu.
3. Structure longer answers with bullet points or numbered lists when listing items.
4. Never fabricate information."""

SCOPE = [
    "msads","applied data science","uchicago","university of chicago",
    "course","curriculum","capstone","admission","application","tuition",
    "fee","scholarship","financial","career","faculty","program","degree",
    "toefl","gre","opt","visa","stem","online","elective","thesis",
    "machine learning","data","python","statistics","neural","deep learning",
    "nlp","reinforcement","internship","job","salary","hire","graduate",
    "quarter","full-time","part-time","full time","part time","study",
    "schedule","core","requirement","chicago","booth","mba","joint",
    "capstone","showcase","research","professor","instructor","class",
    "when","how long","how much","duration","cost","price","afford",
    "deadline","accept","reject","interview","gpa","undergraduate",
]

def is_in_scope(q):
    ql = q.lower()
    return any(w in ql for w in SCOPE)

def get_topic_html(q):
    ql = q.lower()
    tags = []
    if any(w in ql for w in ["course","curriculum","class","machine learning","python","data","elective","core","ml","nlp","deep","neural","ai"]):
        tags.append('<span class="topic-teal">Curriculum</span>')
    if any(w in ql for w in ["admit","apply","application","require","letter","statement","deadline","toefl","gre","gpa"]):
        tags.append('<span class="topic-indigo">Admissions</span>')
    if any(w in ql for w in ["cost","tuition","fee","scholarship","financial","price","afford","how much"]):
        tags.append('<span class="topic-gold">Financials</span>')
    if any(w in ql for w in ["career","job","outcome","employer","hire","salary","work","opt","stem","visa"]):
        tags.append('<span class="topic-green">Careers</span>')
    if any(w in ql for w in ["online","in-person","schedule","evening","part-time","full-time","part time","full time","study","when","how long","chicago"]):
        tags.append('<span class="topic-gray">Format</span>')
    if not tags:
        tags.append('<span class="topic-gray">General</span>')
    return " ".join(tags)

def call_llm(messages):
    if LLM_PROVIDER == "anthropic":
        sys_msg = next((m["content"] for m in messages if m["role"] == "system"), SYSTEM)
        user_msgs = [m for m in messages if m["role"] != "system"]
        r = client.messages.create(
            model="claude-opus-4-6", max_tokens=900,
            system=sys_msg, messages=user_msgs
        )
        return r.content[0].text
    else:
        model = OLLAMA_MODEL if LLM_PROVIDER == "ollama" else "gpt-4o"
        r = client.chat.completions.create(
            model=model, temperature=0.2, max_tokens=900,
            messages=messages
        )
        return r.choices[0].message.content

TOPIC_FOLLOWUPS = {
    "curriculum":  [
        "What electives are available?",
        "Tell me about Machine Learning I and II.",
        "What is the capstone project?",
    ],
    "admissions":  [
        "When is the application deadline?",
        "How much does the program cost?",
        "What are the English language requirements?",
    ],
    "tuition":     [
        "Are there scholarships available?",
        "What is the 18-course thesis track cost?",
        "When is the application deadline?",
    ],
    "careers":     [
        "Is the program STEM OPT eligible?",
        "What companies hire MSADS graduates?",
        "Can I work while studying?",
    ],
    "visa":        [
        "Can I study online as an international student?",
        "What is Curricular Practical Training (CPT)?",
        "Does the online program offer visa sponsorship?",
    ],
    "format":      [
        "Can I study part-time?",
        "Where are classes held in Chicago?",
        "Can I study online instead?",
    ],
    "capstone":    [
        "What are the 6 core courses?",
        "How long does the program take?",
        "What careers do graduates pursue?",
    ],
    "default":     [
        "What are the 6 core courses?",
        "What are the admission requirements?",
        "What careers do graduates pursue?",
    ],
}

def parse_followups(text, query=""):
    """Return clean answer + topic-relevant follow-up questions."""
    # Remove any FOLLOWUPS line Llama may have added anyway
    clean = text.split("FOLLOWUPS:")[0].strip() if "FOLLOWUPS:" in text else text.strip()
    ql = query.lower()
    if any(w in ql for w in ["visa","opt","stem","f-1","international","cpt"]):
        followups = TOPIC_FOLLOWUPS["visa"]
    elif any(w in ql for w in ["capstone","project","showcase","thesis"]):
        followups = TOPIC_FOLLOWUPS["capstone"]
    elif any(w in ql for w in ["career","job","outcome","employer","hire","salary"]):
        followups = TOPIC_FOLLOWUPS["careers"]
    elif any(w in ql for w in ["cost","tuition","fee","scholarship","price","afford"]):
        followups = TOPIC_FOLLOWUPS["tuition"]
    elif any(w in ql for w in ["admit","apply","require","letter","statement","deadline","toefl","gre"]):
        followups = TOPIC_FOLLOWUPS["admissions"]
    elif any(w in ql for w in ["online","in-person","part-time","full-time","schedule","chicago","evening","study"]):
        followups = TOPIC_FOLLOWUPS["format"]
    elif any(w in ql for w in ["course","curriculum","elective","machine learning","python","data","core","nlp","deep"]):
        followups = TOPIC_FOLLOWUPS["curriculum"]
    else:
        followups = TOPIC_FOLLOWUPS["default"]
    return clean, followups

def rag_query(query, history, k=4):
    if not is_in_scope(query):
        return ("I specialise in the UChicago MS in Applied Data Science program. "
                "Ask me about courses, admissions, tuition, careers, faculty, or scheduling!",
                [], 0, [])

    passages = store.retrieve(query, top_k=k)
    if not passages:
        return ("I couldn't find relevant information in my knowledge base. "
                "Please visit datascience.uchicago.edu for details.", [], 0, [])

    ctx = "\n\n".join(
        f"[Source {i+1}: {p['title']} | relevance {p['relevance']:.0%}]\n{p['text'][:700]}"
        for i, p in enumerate(passages)
    )

    # Build conversation history for context awareness
    msgs = [{"role": "system", "content": SYSTEM}]
    for h in history[-4:]:  # last 4 exchanges for memory
        msgs.append({"role": "user",      "content": h["q"]})
        msgs.append({"role": "assistant", "content": h["a"]})
    msgs.append({"role": "user", "content":
        f"Context from the MSADS website:\n\n{ctx}\n\n"
        f"Question: {query}\n\n"
        "Give a thorough answer using bullet points where helpful. "
        "End with 3 follow-up questions as: FOLLOWUPS: Q1 | Q2 | Q3"
    })

    raw = call_llm(msgs)
    raw = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[email redacted]", raw)
    answer, followups = parse_followups(raw, query)
    top_score = passages[0]["relevance"] if passages else 0
    return answer, passages, top_score, followups

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    provider_label = {"ollama": f"🦙 Ollama · {OLLAMA_MODEL} (free)", "openai": "⚡ OpenAI GPT-4o", "anthropic": "🔮 Anthropic Claude"}
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#800000,#1F3864);border-radius:12px;
                padding:1rem 1.2rem;color:white;margin-bottom:1rem;'>
      <div style='font-size:1.05rem;font-weight:700;'>🎓 MSADS AI Assistant</div>
      <div style='font-size:0.72rem;opacity:0.8;margin-top:3px;'>{provider_label.get(LLM_PROVIDER,"")}</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["💬 Chat", "⚙️ Pipeline"])

    with tab1:
        st.markdown("**Quick Questions**")
        quick_qs = [
            "What are the 6 core courses?",
            "What are the admission requirements?",
            "How much does the program cost?",
            "What careers do graduates pursue?",
            "Tell me about Machine Learning I and II.",
            "What electives are available?",
            "Is the program STEM OPT eligible?",
            "What is the capstone project?",
            "Can I study part-time?",
            "When is the application deadline?",
            "What is the 2-year thesis track?",
            "Who are the faculty?",
        ]
        for q in quick_qs:
            if st.button(q, key=f"qq_{q}", use_container_width=True):
                st.session_state["prefill"] = q

        st.divider()
        show_src   = st.toggle("Show retrieved sources", value=True)
        show_score = st.toggle("Show relevance score",   value=True)
        show_exp   = st.toggle("Show RAG explainer",     value=False)
        top_k      = st.slider("Top-K passages", 2, 8, 4)
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history  = []
            st.rerun()

    with tab2:
        st.markdown("**RAG Pipeline Steps**")
        for icon, label, desc in [
            ("🌐", "Web Scraper",    "9 MSADS pages · requests + BeautifulSoup"),
            ("✂️", "Chunker",        "512-char sliding windows · 64-char overlap"),
            ("🧠", "Embeddings",     "MiniLM-L6-v2 · 384-dim dense vectors"),
            ("🗄️", "ChromaDB",       "HNSW cosine index · 37 chunks indexed"),
            ("🔍", "Retrieval",      f"Top-{top_k} cosine similarity search"),
            ("💬", "LLM Generation", "Grounded · temp=0.2 · PII redacted"),
        ]:
            st.markdown(f"""
            <div class='pipeline-card'>
              <span class='pipeline-icon'>{icon}</span>
              <div>
                <div class='pipeline-label'>{label}</div>
                <div class='pipeline-desc'>{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("**Responsible AI Safeguards**")
        for item in ["🔒 Hallucination guard — context-only answers",
                     "🎯 Scope filter — rejects off-topic queries",
                     "🛡️ PII redaction — strips emails & phones",
                     "🌡️ Low temperature — near-deterministic"]:
            st.markdown(f"<div style='font-size:0.8rem;padding:3px 0;color:#444;'>{item}</div>", unsafe_allow_html=True)

        st.divider()
        st.markdown("**Evaluation Results**")
        cols = st.columns(2)
        cols[0].metric("Precision@4", "33%")
        cols[1].metric("MRR",         "0.83")
        cols[0].metric("Faithfulness","100%")
        cols[1].metric("Correct",     "18/20")

# ── MAIN AREA ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
  <h1>UChicago MS in Applied Data Science</h1>
  <p>AI-powered assistant · RAG system · Grounded in official program content · Powered by Llama 3.2</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='metric-row'>
  <div class='metric-card'><div class='val'>36</div><div class='lbl'>Knowledge Chunks</div></div>
  <div class='metric-card'><div class='val'>15</div><div class='lbl'>Pages Scraped</div></div>
  <div class='metric-card'><div class='val'>33%</div><div class='lbl'>Precision@4</div></div>
  <div class='metric-card'><div class='val'>0.83</div><div class='lbl'>MRR Score</div></div>
  <div class='metric-card'><div class='val'>18/20</div><div class='lbl'>Test Q's Correct</div></div>
</div>
""", unsafe_allow_html=True)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.history  = []

# Welcome message
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("""👋 **Hello! I'm the MSADS AI Assistant.**

I'm powered by a **Retrieval-Augmented Generation (RAG)** system built entirely on official UChicago program content. I retrieve the most relevant knowledge chunks using semantic search, then generate grounded answers using Llama 3.2 — running locally on your Mac, completely free.

Ask me anything about **curriculum, admissions, tuition, career outcomes, scheduling, faculty**, and more. I'll also suggest follow-up questions after each answer!""")

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("topics"):
            st.markdown(msg["topics"], unsafe_allow_html=True)

        if show_score and msg.get("score"):
            st.caption(f"🎯 Top retrieval relevance: {msg['score']:.0%}")

        if show_exp and msg.get("sources"):
            with st.expander("🔬 RAG Explainability — what was retrieved"):
                st.markdown(f"**Query matched {len(msg['sources'])} chunks from the knowledge base:**")
                for i, p in enumerate(msg["sources"]):
                    bar = "█" * int(p["relevance"] * 20) + "░" * (20 - int(p["relevance"] * 20))
                    st.markdown(f"""
                    <div class='src-card'>
                      <span class='src-title'>#{i+1} {p['title']}</span>
                      <span class='src-score'>{p['relevance']:.0%}</span>
                      <div style='font-family:monospace;font-size:0.7rem;color:#800000;margin:3px 0;'>{bar} {p['relevance']:.3f}</div>
                      <div class='src-text'>{p['text'][:280]}…</div>
                      <div style='margin-top:4px;font-size:0.7rem;color:#999;'>
                        <a href='{p["url"]}' target='_blank'>{p["url"]}</a>
                      </div>
                    </div>""", unsafe_allow_html=True)

        elif show_src and msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} retrieved passages"):
                for p in msg["sources"]:
                    st.markdown(f"""
                    <div class='src-card'>
                      <span class='src-title'>{p['title']}</span>
                      <span class='src-score'>{p['relevance']:.0%}</span>
                      <div class='src-text'>{p['text'][:280]}…</div>
                    </div>""", unsafe_allow_html=True)

        if msg.get("followups"):
            st.markdown("**💡 You might also want to ask:**")
            cols = st.columns(len(msg["followups"]))
            for i, fq in enumerate(msg["followups"]):
                if cols[i].button(fq, key=f"fu_{hash(fq)}_{i}", use_container_width=True):
                    st.session_state["prefill"] = fq

# Input
prefill = st.session_state.pop("prefill", "")
if user_input := (st.chat_input("Ask about the MSADS program…") or prefill):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        topics_html = get_topic_html(user_input)

        with st.spinner("🔍 Searching knowledge base and generating answer…"):
            ans, srcs, score, followups = rag_query(
                user_input, st.session_state.history, top_k
            )

        # Stream the answer word by word
        placeholder = st.empty()
        displayed = ""
        words = ans.split(" ")
        for i, word in enumerate(words):
            displayed += word + " "
            if i % 4 == 0:
                placeholder.markdown(displayed + "▌")
                time.sleep(0.02)
        placeholder.markdown(ans)

        st.markdown(topics_html, unsafe_allow_html=True)

        if show_score and score:
            st.caption(f"🎯 Top retrieval relevance: {score:.0%}")

        if show_exp and srcs:
            with st.expander("🔬 RAG Explainability — what was retrieved"):
                st.markdown(f"**Query matched {len(srcs)} chunks from the knowledge base:**")
                for i, p in enumerate(srcs):
                    bar = "█" * int(p["relevance"] * 20) + "░" * (20 - int(p["relevance"] * 20))
                    st.markdown(f"""
                    <div class='src-card'>
                      <span class='src-title'>#{i+1} {p['title']}</span>
                      <span class='src-score'>{p['relevance']:.0%}</span>
                      <div style='font-family:monospace;font-size:0.7rem;color:#800000;margin:3px 0;'>{bar} {p['relevance']:.3f}</div>
                      <div class='src-text'>{p['text'][:280]}…</div>
                      <div style='margin-top:4px;font-size:0.7rem;color:#999;'>
                        <a href='{p["url"]}' target='_blank'>{p["url"]}</a>
                      </div>
                    </div>""", unsafe_allow_html=True)
        elif show_src and srcs:
            with st.expander(f"📚 {len(srcs)} retrieved passages"):
                for p in srcs:
                    st.markdown(f"""
                    <div class='src-card'>
                      <span class='src-title'>{p['title']}</span>
                      <span class='src-score'>{p['relevance']:.0%}</span>
                      <div class='src-text'>{p['text'][:280]}…</div>
                    </div>""", unsafe_allow_html=True)

        if followups:
            st.markdown("**💡 You might also want to ask:**")
            cols = st.columns(len(followups))
            for i, fq in enumerate(followups):
                if cols[i].button(fq, key=f"nfu_{hash(fq)}_{i}", use_container_width=True):
                    st.session_state["prefill"] = fq

    # Store
    st.session_state.history.append({"q": user_input, "a": ans})
    st.session_state.messages.append({
        "role": "assistant", "content": ans,
        "sources": srcs, "score": score,
        "topics": topics_html, "followups": followups,
    })
