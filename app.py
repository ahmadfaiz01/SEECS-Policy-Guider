"""
app.py
The main Streamlit interface for our SEECS Policy QA System.
Uses native Streamlit components for a clean, universally pleasing UI.
"""
import sys
import os
import time
import json
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

load_dotenv()

from retrieval.pipeline import RetrievalPipeline
from generation.answer_gen import AnswerGenerator
from extensions.pagerank import HandbookPageRank

INDEX_DIR   = ROOT / "data" / "indexes"
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.jsonl"
RESULTS_DIR = ROOT / "experiments" / "results"
FIGURES_DIR = ROOT / "experiments" / "figures"
CHART_COLORS = ["#2563eb", "#0284c7", "#7c3aed", "#dc2626"]

st.set_page_config(
    page_title="SEECS Policy QA",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: #f4f8fb;
        color: #17202a;
    }
    .block-container {
        padding-top: 4rem;
        max-width: 1500px;
    }
    [data-testid="stSidebar"] {
        background: #f1f6fa;
        border-right: 1px solid #d2dce6;
    }
    [data-testid="stSidebar"] * {
        color: #17202a;
    }
    div[data-testid="stMetric"] {
        background: #eef7ff;
        border: 1px solid #b8d9f4;
        border-radius: 8px;
        padding: 0.55rem 0.7rem;
    }
    div[data-testid="stMetric"] * {
        color: #17202a;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #eef7ff;
        border-color: #b8d9f4;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.08);
    }
    h1, h2, h3, p, span, label {
        color: #17202a;
    }
    .stButton > button {
        border-radius: 7px;
        border: 1px solid #c0d0e0;
        background: #eef7ff;
        color: #17202a;
        min-height: 2.7rem;
        white-space: normal;
    }
    .stButton > button:hover {
        border-color: #2f5f8f;
        color: #17202a;
        background: #dff0ff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 0.9rem;
    }
    div[data-testid="stExpander"] {
        background: #f4faff;
        border-color: #b8d9f4;
    }
    .hero-band {
        background: #eef7ff;
        border: 1px solid #b8d9f4;
        border-radius: 10px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1.2rem;
        color: #17202a;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.08);
    }
    .hero-band h1 {
        color: #17202a;
        margin: 0;
        font-size: 2rem;
    }
    .hero-band p {
        color: #17324d;
        margin: 0.35rem 0 0 0;
    }
    .note-box {
        background: #eef7ff;
        border: 1px solid #b8d9f4;
        border-left: 5px solid #2563eb;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin: 0.8rem 0 1rem 0;
    }
    .note-box p {
        margin: 0.25rem 0;
        color: #17324d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cached Loading ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initializing retrieval engines...")
def load_pipeline():
    if not INDEX_DIR.exists() or not CHUNKS_FILE.exists():
        with st.spinner("First time setup: Building indexes from PDFs (this takes ~10 seconds)..."):
            import subprocess
            try:
                subprocess.run([sys.executable, "build_index.py"], check=True)
            except Exception as e:
                st.error(f"Failed to build indexes automatically: {e}")
                return None
                
    if INDEX_DIR.exists() and CHUNKS_FILE.exists():
        return RetrievalPipeline(str(INDEX_DIR), str(CHUNKS_FILE))
    return None

@st.cache_resource(show_spinner="Initializing network graphs...")
def load_pagerank():
    path = INDEX_DIR / "pagerank.pkl"
    if not path.exists():
        return None
    return HandbookPageRank.load(str(path))

@st.cache_resource(show_spinner="Connecting to LLM API...")
def load_generator():
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return None
    try:
        return AnswerGenerator(api_key=key)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def get_cached_answer(query: str, chunks_json: str, method_key: str):
    generator = load_generator()
    if not generator:
        return None
    chunks = json.loads(chunks_json)
    return generator.generate(query, chunks)

# ── Sidebar ───────────────────────────────────────────────────────────────────

METHOD_LABELS = {
    "hybrid_lsh": "Hybrid LSH",
    "minhash": "MinHash LSH",
    "simhash": "SimHash",
    "tfidf": "TF-IDF Baseline",
}


def style_chart(fig):
    fig.update_layout(
        paper_bgcolor="#f4f8fb",
        plot_bgcolor="#ffffff",
        font=dict(color="#17202a", size=13),
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="",
        title_font=dict(size=18, color="#17202a"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#d9e8f5", zeroline=False)
    return fig


def render_sidebar(pipeline=None):
    with st.sidebar:
        st.header("Configuration")
        top_k = st.slider("Documents to Retrieve", 3, 10, 5)
        
        # Answer strategy is now determined by the active tab
        answer_method = "hybrid_lsh" 
        
        use_pagerank = st.toggle("Apply PageRank Boost", value=True,
                                 help="Boosts chunks from heavily referenced sections.")
        show_scores = st.toggle("Show PageRank Score", value=True)
        
        st.divider()
        st.header("Test Queries")
        
        samples = [
            "What is the minimum CGPA for graduation?",
            "What happens if a student fails a course?",
            "What is the attendance policy?",
            "How many times can a course be repeated?",
            "What is the Dean's Honor List requirement?",
            "What is the procedure for dropping a course?",
        ]
        for s in samples:
            if st.button(s, width="stretch"):
                st.session_state["query_input"] = s
                
        st.divider()
        st.caption("SEECS Policy Retrieval System")
    return top_k, answer_method, use_pagerank, show_scores

# ── UI Helpers ────────────────────────────────────────────────────────────────

def pagerank_details(chunk: dict, pr_model):
    if not pr_model or not pr_model.scores:
        return 0.0, 0.0
    section = chunk.get("section", "General")
    raw = pr_model.score(section)
    max_pr = max(pr_model.scores.values()) or 1.0
    return raw, raw / max_pr


def render_chunk_card(
    chunk: dict,
    score: float,
    show_scores: bool,
    rank: int,
    pr_model=None,
    retrieval_score: float = None,
):
    """Renders a chunk using standard Streamlit containers."""
    source = chunk.get("source", "Unknown").replace(".pdf", "")
    sp = chunk.get("start_page", "?")
    ep = chunk.get("end_page", sp)
    pages = f"Pg. {sp}" if sp == ep else f"Pg. {sp}-{ep}"
    section = chunk.get("section", "General").strip()
    text = chunk.get("text", "")
    
    with st.container(border=True):
        st.markdown(f"**[{rank}] {source}**")
        st.caption(f"{section} | {pages}")
        if show_scores:
            raw_pr, _ = pagerank_details(chunk, pr_model)
            st.metric("PageRank", f"{raw_pr:.5f}")
        
        # Display the text inside an expander if it's long, or just plain text
        if len(text) > 250:
            st.write(text[:250] + "...")
            with st.expander("Read full excerpt"):
                st.write(text)
        else:
            st.write(text)

def render_method_column(label: str, method_key: str, result, show_scores: bool, pr_model, use_pagerank: bool, query: str, top_k: int):
    """Renders a column for a specific retrieval algorithm."""
    st.subheader(label)
    
    st.metric("Latency", f"{result.latency_ms:.1f} ms")
    
    st.divider()

    chunks = result.chunks
    scores = result.scores
    original_scores = {c.get("chunk_id"): s for c, s in zip(chunks, scores)}

    if use_pagerank and pr_model and pr_model.scores:
        boosted = pr_model.boost(chunks, scores, alpha=0.25)
        chunks = [b[0] for b in boosted]
        scores = [b[1] for b in boosted]

    st.subheader("Synthesized Answer")
    generator = load_generator()
    if generator is None:
        st.warning("Answer synthesis is disabled. GROQ_API_KEY is not configured.")
    else:
        with st.spinner(f"Synthesizing response for {label}..."):
            top_chunks = chunks[:top_k]
            chunks_json = json.dumps([{"text": c.get("text", ""), "source": c.get("source", ""), "start_page": c.get("start_page", ""), "end_page": c.get("end_page", ""), "section": c.get("section", "")} for c in top_chunks])
            ans = get_cached_answer(query, chunks_json, method_key)
            if ans:
<<<<<<< HEAD
                st.markdown(f"**Policy Answer:**\n\n{ans['answer']}")
=======
                st.info(ans["answer"])
>>>>>>> origin/ahmad
                st.write("**References:**")
                for s in ans["sources"]:
                    st.caption(f"- {s}")

    st.divider()
    st.subheader("Retrieved Source Documents")

    if not chunks:
        st.info("No matching documents found.")
    else:
        for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
            retrieval_score = original_scores.get(chunk.get("chunk_id"), score)
            render_chunk_card(chunk, score, show_scores, i, pr_model, retrieval_score)

def apply_pagerank_boost(chunks: list, scores: list, pr_model, use_pagerank: bool, alpha: float = 0.25):
    if use_pagerank and pr_model and pr_model.scores:
        boosted = pr_model.boost(chunks, scores, alpha=alpha)
        return [b[0] for b in boosted], [b[1] for b in boosted]
    return chunks, scores

# ── Tab 1: Main QA Interface ──────────────────────────────────────────────────

def tab_qa(pipeline, pr_model, generator, top_k, answer_method, use_pagerank, show_scores):
    st.markdown(
        """
        <div class="hero-band">
            <h1 style="color: #17202a !important; margin: 0; font-size: 2rem;">SEECS Policy Retrieval System</h1>
            <p>Retrieval-first QA over UG and PG handbooks. Answer changes when strategy changes because each strategy gives different excerpts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if pipeline is None:
        st.error("System indexes not found. Please execute `python build_index.py`.")
        return

    query = st.text_input(
        "Enter your query:",
        placeholder="e.g. What is the minimum CGPA requirement for graduation?",
        key="query_input",
    )

    if not query:
        return

    with st.spinner("Searching indexes..."):
        all_results = pipeline.query_all(query, top_k=top_k)

    # ── Document Retrieval Comparison ──
    st.subheader("Retrieval & Synthesis Comparison")
    
    method_order = list(METHOD_LABELS.items())
    available = [(key, label) for key, label in method_order if key in all_results]
    if available:
        method_tabs = st.tabs([label for _, label in available])
        for tab, (key, label) in zip(method_tabs, available):
            with tab:
                render_method_column(label, key, all_results[key], show_scores, pr_model, use_pagerank, query, top_k)

# ── Tab 2: Analytics Dashboard ────────────────────────────────────────────────

def tab_analytics():
    st.header("Experimental Results & Analysis")
    st.caption("Benchmark CSVs compare exact baseline, approximate retrieval, parameter sensitivity, and corpus growth.")

    exp1_path = RESULTS_DIR / "exp1_comparison.csv"
    if not exp1_path.exists():
        st.info("Results not found. Execute `python experiments/benchmark.py` to generate analytics.")
        return

    st.subheader("Exact vs Approximate Retrieval")
    df1 = pd.read_csv(exp1_path)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="note-box">
                <p><b>Latency:</b> How fast each algorithm searches the entire document corpus (lower is better).</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        means = {
            "TF-IDF":       df1["tfidf_latency_ms"].mean(),
            "Hybrid LSH":   df1["hybrid_lsh_latency_ms"].mean() if "hybrid_lsh_latency_ms" in df1 else None,
            "MinHash+LSH":  df1["minhash_latency_ms"].mean(),
            "SimHash":      df1["simhash_latency_ms"].mean(),
        }
        means = {k: v for k, v in means.items() if v is not None}
        fig = px.bar(x=list(means.keys()), y=list(means.values()),
                     labels={"x": "Method", "y": "Avg Latency (ms)"},
                     title="Query Latency Comparison",
                     color=list(means.keys()))
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown(
            """
            <div class="note-box">
                <p><b>Precision@5:</b> out of top 5 chunks returned by a method, how many also appear in the TF-IDF top 5 baseline.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        prec_means = {
            "TF-IDF (Baseline)": 1.0,
            "Hybrid LSH":         df1["hybrid_lsh_precision5"].mean() if "hybrid_lsh_precision5" in df1 else None,
            "MinHash+LSH":       df1["minhash_precision5"].mean(),
            "SimHash":           df1["simhash_precision5"].mean(),
        }
        prec_means = {k: v for k, v in prec_means.items() if v is not None}
        fig = px.bar(x=list(prec_means.keys()), y=list(prec_means.values()),
                     labels={"x": "Method", "y": "Precision@5"},
                     title="Retrieval Precision@5",
                     color=list(prec_means.keys()))
        fig.update_layout(yaxis_range=[0, 1.1])
        st.plotly_chart(fig, width="stretch")

    exp3_path = RESULTS_DIR / "exp3_scalability.csv"
    if exp3_path.exists():
        st.subheader("Scalability Analysis")
        st.markdown(
            """
            <div class="note-box">
                <p><b>Scalability:</b> Compares how search time increases as the number of chunks grows.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        df3 = pd.read_csv(exp3_path)
        scalability_methods = [
            col for col in [
                "tfidf_latency_ms",
                "hybrid_lsh_latency_ms",
                "minhash_latency_ms",
                "simhash_latency_ms",
            ]
            if col in df3
        ]
        fig = px.line(df3, x="num_chunks",
                      y=scalability_methods,
                      markers=True, log_x=True, log_y=True,
                      title="Query Latency vs Corpus Size (Log-Log Scale)",
                      labels={"num_chunks": "Corpus Size (Chunks)", "value": "Latency (ms)"})
        st.plotly_chart(fig, width="stretch")

    llm_eval_path = RESULTS_DIR / "llm_accuracy_benchmark.csv"
    if llm_eval_path.exists():
        st.subheader("LLM-Judged Answer Accuracy")
        st.markdown(
            """
            <div class="note-box">
                <p><b>LLM Correctness:</b> Automated judge rates answers 0-5 based on how accurate they are.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        df_eval = pd.read_csv(llm_eval_path)
        summary = (
            df_eval.groupby("method_label", as_index=False)
            .agg(
                correctness=("correctness", "mean"),
            )
            .round(2)
        )
        fig = px.bar(
            summary,
            x="method_label",
            y="correctness",
            labels={"method_label": "Method", "correctness": "Avg Correctness (0-5)"},
            title="LLM-Judged Correctness by Strategy",
            color="method_label",
        )
        fig.update_layout(yaxis_range=[0, 5])
        st.plotly_chart(fig, width="stretch")

# ── Tab 3: PageRank Viz ───────────────────────────────────────────────────────

def tab_pagerank(pr_model):
    st.header("Handbook Section PageRank Analysis")
    st.caption("Section authority is computed from detected handbook cross-references and used as a retrieval boost.")

    if pr_model is None:
        st.warning("PageRank model not initialized.")
        return

    top_n = st.slider("Number of Sections to Display", 5, 30, 15)
    top_sections = pr_model.top_sections(top_n)

    if not top_sections:
        st.info("No explicit cross-references detected within the corpus.")
        return

    labels = [s[:45] + "..." if len(s) > 45 else s for s, _ in top_sections]
    scores = [sc for _, sc in top_sections]

    g = pr_model.graph
    c1, c2, c3 = st.columns(3)
    c1.metric("Sections", f"{g.number_of_nodes() if g else len(pr_model.scores)}")
    c2.metric("Cross References", f"{g.number_of_edges() if g else 0}")
    c3.metric("Max PageRank", f"{max(pr_model.scores.values()):.5f}")

    st.markdown(
        """
        <div class="note-box">
            <p><b>PageRank:</b> Sections heavily cross-referenced by other sections score higher.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = px.bar(x=scores, y=labels, orientation="h",
                 labels={"x": "Computed PageRank Score", "y": "Document Section"},
                 title="Highest Ranked Policy Sections",
                 color=scores, color_continuous_scale="Blues")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")

    if g and g.number_of_nodes() > 0:
        st.subheader("Network Topology")
        st.markdown(
            """
            <div class="note-box">
                <p><b>Network Topology:</b> Visualizes direct cross-references between different handbook sections.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        import networkx as nx
        try:
            pos = nx.spring_layout(g, seed=42)
            edge_x, edge_y = [], []
            for edge in g.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')

            node_x, node_y, node_text, node_color = [], [], [], []
            for node in g.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                sc = pr_model.score(node)
                node_text.append(f"{node}<br>Score: {sc:.4f}")
                node_color.append(sc)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hovertext=node_text,
                hoverinfo='text',
                marker=dict(
                    showscale=True,
                    colorscale='Blues',
                    reversescale=False,
                    color=node_color,
                    size=12,
                    colorbar=dict(thickness=15, title='Score'),
                    line_width=2))

            fig_net = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=20),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                     )
            st.plotly_chart(fig_net, width="stretch")
        except Exception as e:
            st.error(f"Could not render network graph: {e}")

    st.markdown("**Top PageRank Sections**")
    header = st.columns([5, 1, 1])
    header[0].caption("Section")
    header[1].caption("PageRank")
    header[2].caption("Normalized")
    max_score = max(pr_model.scores.values()) or 1.0
    for section, score in top_sections:
        row = st.columns([5, 1, 1])
        row[0].write(section)
        row[1].write(f"{score:.6f}")
        row[2].write(f"{score / max_score:.4f}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pipeline  = load_pipeline()
    pr_model  = load_pagerank()
    generator = load_generator()

    top_k, answer_method, use_pagerank, show_scores = render_sidebar(pipeline)

    # The tabs are defined here!
    tab1, tab2, tab3 = st.tabs(["Search & Retrieval", "Performance Analytics", "Knowledge Graph"])

    with tab1:
        tab_qa(pipeline, pr_model, generator, top_k, answer_method, use_pagerank, show_scores)
    with tab2:
        tab_analytics()
    with tab3:
        tab_pagerank(pr_model)

if __name__ == "__main__":
    main()
