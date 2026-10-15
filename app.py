import os
import ast
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
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
    st.subheader("Side-by-Side Mode Comparison: Sparse vs Dense")
    st.caption("Directly contrast Sparse Keyword (TF-IDF) against Dense Semantic (all-MiniLM-L6-v2) on the same input.")

    comp_query = st.text_input(
        "Enter a test query for comparison:",
        value="Chef wants to find shortest path between cities avoiding tolls",
        key="comp_query_input"
    )
    k_comp = st.slider("Results to Compare", 3, 15, 6, key="k_comp_slider")

    if comp_query.strip():
        kw_scores = searcher.scores(comp_query, mode="keyword")
        sem_scores = searcher.scores(comp_query, mode="semantic")

        kw_idx = np.argsort(-kw_scores)[:k_comp]
        sem_idx = np.argsort(-sem_scores)[:k_comp]

        kw_df = df.iloc[kw_idx].copy(); kw_df["score"] = kw_scores[kw_idx]
        sem_df = df.iloc[sem_idx].copy(); sem_df["score"] = sem_scores[sem_idx]

        kw_codes = set(kw_df["qcode"])
        sem_codes = set(sem_df["qcode"])
        common = kw_codes & sem_codes
        jaccard = len(common) / len(kw_codes | sem_codes) if (kw_codes | sem_codes) else 0.0

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Shared Problems in Top-K", f"{len(common)} / {k_comp}")
        with m_col2:
            st.metric("Jaccard Similarity Overlap", f"{jaccard:.2f}")
        with m_col3:
            st.metric("Novel Semantic Discoveries", f"{len(sem_codes - kw_codes)}")

        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown("#### 🔤 Sparse Keyword (TF-IDF)")
            st.caption("Matches exact word tokens (often distracted by story words like *Chef*, *city*, *game*).")
            for rank, (_, row) in enumerate(kw_df.iterrows(), 1):
                is_shared = row["qcode"] in common
                border_color = "#3B82F6" if is_shared else "#CBD5E1"
                shared_tag = " <span style='color: #2563EB; font-size: 0.8rem; font-weight: 600;'>[In Both]</span>" if is_shared else ""
                diff = str(row.get("difficulty", "medium")).lower()
                badge_class = f"badge-{diff}" if diff in DIFF_ORDER else "badge-medium"
                link = f"https://www.codechef.com{row['link']}" if str(row.get("link", "")).startswith("/") else f"https://www.codechef.com/problems/{row.get('qcode', '')}"
                
                st.markdown(f"""
                <div style="border: 2px solid {border_color}; border-radius: 6px; padding: 10px; margin-bottom: 8px; background: #FAFAFA;">
                    <div style="font-weight: 600;">#{rank}. <a href="{link}" target="_blank">{row['title']}</a> ({row['qcode']}){shared_tag}</div>
                    <div style="font-size: 0.82rem; color: #64748B; margin-top: 4px;">
                        <span class="{badge_class}">{diff.upper()}</span> | Score: {row['score']:.3f} | Tags: {', '.join(row['tags'][:3]) if isinstance(row['tags'], list) and row['tags'] else 'None'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with c_right:
            st.markdown("#### 🧠 Dense Semantic (MiniLM-L6-v2)")
            st.caption("Matches conceptual problem meaning and algorithm abstraction (even with different story words).")
            for rank, (_, row) in enumerate(sem_df.iterrows(), 1):
                is_shared = row["qcode"] in common
                border_color = "#3B82F6" if is_shared else "#10B981"
                shared_tag = " <span style='color: #2563EB; font-size: 0.8rem; font-weight: 600;'>[In Both]</span>" if is_shared else " <span style='color: #059669; font-size: 0.8rem; font-weight: 600;'>[Semantic Match]</span>"
                diff = str(row.get("difficulty", "medium")).lower()
                badge_class = f"badge-{diff}" if diff in DIFF_ORDER else "badge-medium"
                link = f"https://www.codechef.com{row['link']}" if str(row.get("link", "")).startswith("/") else f"https://www.codechef.com/problems/{row.get('qcode', '')}"

                st.markdown(f"""
                <div style="border: 2px solid {border_color}; border-radius: 6px; padding: 10px; margin-bottom: 8px; background: #F0FDF4;">
                    <div style="font-weight: 600;">#{rank}. <a href="{link}" target="_blank">{row['title']}</a> ({row['qcode']}){shared_tag}</div>
                    <div style="font-size: 0.82rem; color: #64748B; margin-top: 4px;">
                        <span class="{badge_class}">{diff.upper()}</span> | Score: {row['score']:.3f} | Tags: {', '.join(row['tags'][:3]) if isinstance(row['tags'], list) and row['tags'] else 'None'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ----------------- TAB 3: Interactive Clusters -----------------
with tabs[2]:
    st.subheader("2D Semantic Problem Map & Cluster Exploration")
    st.caption("Nonlinear 2D projection (UMAP) of dense problem embeddings with discovered thematic clusters.")
    
    if coords is not None and "cluster_label" in df.columns:
        plot_df = df.copy()
        plot_df["x"] = coords[:, 0]
        plot_df["y"] = coords[:, 1]
        plot_df["tags_display"] = plot_df["tags"].apply(lambda t: ", ".join(t[:5]) if isinstance(t, list) else "")
        
        c_filter1, c_filter2, c_filter3 = st.columns([2, 2, 2])
        with c_filter1:
            color_choice = st.selectbox("Color By", ["Cluster", "Difficulty"])
        with c_filter2:
            cluster_opts = ["All Clusters"] + sorted(plot_df["cluster_label"].dropna().unique().tolist())
            selected_cluster = st.selectbox("Filter Cluster", cluster_opts)
        with c_filter3:
            diff_opts = ["All Difficulties"] + [d.capitalize() for d in DIFF_ORDER]
            selected_diff = st.selectbox("Filter Difficulty", diff_opts)
            
        filtered_plot = plot_df.copy()
        if selected_cluster != "All Clusters":
            filtered_plot = filtered_plot[filtered_plot["cluster_label"] == selected_cluster]
        if selected_diff != "All Difficulties":
            filtered_plot = filtered_plot[filtered_plot["difficulty"].astype(str).str.lower().str.strip() == selected_diff.lower()]
            
        color_col = "cluster_label" if color_choice == "Cluster" else "difficulty"
        
        fig = go.Figure()
        for cat_name, sub in filtered_plot.groupby(color_col):
            hover_texts = [
                f"<b>{row['title']}</b> ({row['qcode']})<br>"
                f"Difficulty: {str(row['difficulty']).upper()}<br>"
                f"Cluster: {row.get('cluster_label', 'N/A')}<br>"
                f"Tags: {', '.join(row['tags'][:4]) if isinstance(row['tags'], list) and row['tags'] else 'None'}"
                for _, row in sub.iterrows()
            ]
            fig.add_trace(go.Scatter(
                x=sub["x"],
                y=sub["y"],
                mode="markers",
                name=str(cat_name),
                text=hover_texts,
                hoverinfo="text",
                marker=dict(size=7, opacity=0.82)
            ))
            
        fig.update_layout(
            title=f"2D Problem Semantic Space ({len(filtered_plot):,} problems shown)",
            height=620,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=40, b=80),
            xaxis=dict(title="UMAP Component 1", showgrid=True),
            yaxis=dict(title="UMAP Component 2", showgrid=True)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster Summary Table
        summary_path = "out/cluster_summary.csv"
        if os.path.exists(summary_path):
            st.markdown("#### 📊 Algorithmic Cluster Profiles & Exemplar Medoids")
            sum_df = pd.read_csv(summary_path)
            st.dataframe(
                sum_df[["cluster_id", "top_topic_tags", "top_tfidf_terms", "size", "exemplar_1", "exemplar_2"]],
                column_config={
                    "cluster_id": "Cluster #",
                    "top_topic_tags": "Dominant Tags",
                    "top_tfidf_terms": "Characteristic Terms",
                    "size": "Problem Count",
                    "exemplar_1": "Medoid Problem 1",
                    "exemplar_2": "Medoid Problem 2"
                },
                use_container_width=True,
                hide_index=True
            )
            
        # Hidden Matches Expander
        hidden_path = "out/hidden_matches.csv"
        if os.path.exists(hidden_path):
            with st.expander("🔍 Discovered 'Hidden Matches' (High Semantic Similarity, Low Lexical Overlap)"):
                st.caption("Pairs of problems with completely different story words that share the exact same underlying algorithm.")
                h_df = pd.read_csv(hidden_path)
                st.dataframe(
                    h_df[["prob1_code", "prob2_code", "prob1_title", "prob2_title", "semantic_sim", "keyword_sim", "explanation"]].head(15),
                    column_config={
                        "prob1_code": "Query Code",
                        "prob2_code": "Match Code",
                        "prob1_title": "Query Title",
                        "prob2_title": "Match Title",
                        "semantic_sim": "Dense Sim (MiniLM)",
                        "keyword_sim": "Sparse Sim (TF-IDF)",
                        "explanation": "Algorithmic Rationale"
                    },
                    use_container_width=True,
                    hide_index=True
                )
    else:
        st.info("2D coordinates not available. Run `python dejacode.py --data data/questions.csv` to generate projection.")

# ----------------- TAB 4: Problem Recommendations -----------------
with tabs[3]:
    st.subheader("Adaptive Concept & Difficulty Progression")
    st.caption("Input solved problems to receive personalized concept-aligned practice recommendations.")
