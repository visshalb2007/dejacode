import os
import ast
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from dejacode import Searcher, recommend_next, attach_solution_stats, get_code_snippet, DIFF_ORDER

st.set_page_config(
    page_title="Déjà Code | Competitive Programming Search & Recommendations",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
    .badge-beginner { background-color: #DEF7EC; color: #03543F; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
    .badge-easy { background-color: #E1EFFE; color: #1E429F; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
    .badge-medium { background-color: #FEF08A; color: #713F12; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
    .badge-hard { background-color: #FEE2E2; color: #991B1B; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
    .badge-challenge { background-color: #F3E8FF; color: #6B21A8; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
    .card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading problem index and models...")
def load_app_data():
    data_path = "out/problems_clustered.csv"
    emb_path = "out/emb.npy"
    coords_path = "out/coords_2d.npy"
    if not (os.path.exists(data_path) and os.path.exists(emb_path)):
        return None, None, None, None
    
    df = pd.read_csv(data_path)
    def safe_parse(v):
        if isinstance(v, list): return v
        try:
            val = ast.literal_eval(str(v))
            return val if isinstance(val, list) else []
        except Exception:
            return []
    df["tags"] = df["tags"].map(safe_parse)
    df["doc"] = df["title"].astype(str) + ". " + df["text"].astype(str)
    
    # Attach acceptance stats if available
    df = attach_solution_stats(df, "data/solutions.csv")
    
    emb = np.load(emb_path)
    coords = np.load(coords_path) if os.path.exists(coords_path) else None
    searcher = Searcher(df, emb)
    return df, emb, coords, searcher

df, emb, coords, searcher = load_app_data()

# Sidebar
with st.sidebar:
    st.markdown("### ⚡ **Déjà Code**")
    st.caption("Competitive Programming Semantic Search & Recommendations")
    st.divider()
    if df is not None:
        st.metric("Problems Indexed", f"{len(df):,}")
        clusters_count = df["cluster"].nunique() if "cluster" in df.columns else 0
        st.metric("Concept Clusters", f"{clusters_count}")
        st.metric("Total Submissions Indexed", f"{df['total_submissions'].sum():,}")
    st.divider()
    st.markdown("Built with **Sentence-Transformers**, **scikit-learn**, **UMAP**, and **Streamlit**.")
    st.markdown("[GitHub Repository](https://github.com/visshalb2007/dejacode)")

st.markdown('<div class="main-header">⚡ Déjà Code</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Algorithmic semantic search, topic clustering, and difficulty progression for CodeChef problems</div>', unsafe_allow_html=True)

if searcher is None:
    st.error("Index artifacts not found in `out/`. Please run `python dejacode.py --data data/questions.csv --k 12` first.")
    st.stop()

tabs = st.tabs([
    "🔍 Search & Lookup",
    "⚖️ Compare Search Modes",
    "🌌 Interactive Clusters",
    "🎯 Problem Recommendations"
])

# ----------------- TAB 1: Search & Lookup -----------------
with tabs[0]:
    st.subheader("Semantic & Hybrid Problem Search")
    
    sample_queries = [
        "-- Select an example query or type your own below --",
        "Shortest path in a weighted directed acyclic graph with obstacles",
        "Given an array find maximum subarray sum with at most k elements",
        "Number of ways to tile a grid using dominoes and trominoes dynamic programming",
        "Find the longest common subsequence of two strings",
        "Count inversions in an array using merge sort or Fenwick tree",
        "Nim game with piles of stones game theory optimal strategy"
    ]
    selected_sample = st.selectbox("Preset Example Queries", sample_queries)
    default_q = "" if selected_sample.startswith("--") else selected_sample
    
    col_q, col_opt = st.columns([3, 1])
    with col_q:
        q = st.text_area("Problem description, story, or algorithm keywords:", value=default_q, height=100, placeholder="Paste a problem statement or describe the underlying algorithmic task...")
    
    with col_opt:
        mode = st.radio("Search Mode", ["hybrid", "semantic", "keyword"], index=0)
        alpha = 0.65
        if mode == "hybrid":
            alpha = st.slider("Dense / Sparse Weight (α)", 0.0, 1.0, 0.65, 0.05, help="1.0 = Pure Semantic Dense, 0.0 = Pure Sparse Keyword")
        k_results = st.slider("Max Results", 3, 20, 5)
        diff_filter = st.selectbox("Difficulty Filter", ["All"] + DIFF_ORDER)

    if q.strip():
        scores = searcher.scores(q, mode=mode, alpha=alpha)
        ranked_idx = np.argsort(-scores)
        
        results_df = df.iloc[ranked_idx].copy()
        results_df["score"] = scores[ranked_idx]
        
        if diff_filter != "All":
            results_df = results_df[results_df["difficulty"].astype(str).str.lower().str.strip() == diff_filter.lower()]
        
        top_k = results_df.head(k_results)
        
        st.markdown(f"#### Top {len(top_k)} Matching Problems (`{mode}` mode)")
        
        for _, row in top_k.iterrows():
            diff = str(row.get("difficulty", "medium")).lower()
            badge_class = f"badge-{diff}" if diff in DIFF_ORDER else "badge-medium"
            codechef_link = f"https://www.codechef.com{row['link']}" if str(row.get("link", "")).startswith("/") else f"https://www.codechef.com/problems/{row.get('qcode', '')}"
            tags_str = ", ".join(row["tags"]) if isinstance(row["tags"], list) and row["tags"] else "None"
            
            with st.container():
                st.markdown(f"""
                <div class="card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #0F172A;">
                            <a href="{codechef_link}" target="_blank" style="text-decoration: none; color: #2563EB;">{row.get('title', 'Unknown')}</a>
                            <span style="font-size: 0.85rem; color: #64748B; font-weight: normal;">({row.get('qcode', 'N/A')})</span>
                        </h4>
                        <div>
                            <span class="{badge_class}">{diff.upper()}</span>
                            <span style="margin-left: 8px; font-weight: 600; font-size: 0.85rem; color: #475569;">Score: {row['score']:.3f}</span>
                        </div>
                    </div>
                    <div style="margin-top: 8px; font-size: 0.88rem; color: #334155;">
                        <strong>Tags:</strong> {tags_str} | 
                        <strong>Cluster:</strong> {row.get('cluster_label', 'N/A')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("Preview Problem Statement"):
                    st.write(row.get("text", "No statement text available."))

# ----------------- TAB 2: Compare Search Modes -----------------
with tabs[1]:
    st.subheader("Side-by-Side Mode Comparison")
    st.caption("Inspect how Sparse Keyword (TF-IDF) vs Dense Semantic (MiniLM) rank problems on the exact same query.")

# ----------------- TAB 3: Interactive Clusters -----------------
with tabs[2]:
    st.subheader("2D Semantic Problem Map")
    st.caption("Explore nonlinear 2D projections of competitive programming problems and discovered algorithmic clusters.")

# ----------------- TAB 4: Problem Recommendations -----------------
with tabs[3]:
    st.subheader("Adaptive Concept & Difficulty Progression")
    st.caption("Input solved problems to receive personalized concept-aligned practice recommendations.")
