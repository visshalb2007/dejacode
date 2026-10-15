import os
import ast
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dejacode import Searcher, recommend_next, attach_solution_stats, get_code_snippet, DIFF_ORDER

st.set_page_config(
    page_title="Deja Code | Competitive Programming Search & Recommendations",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Neomorphism (Soft UI) Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global Soft UI Background & Typography */
    .stApp {
        background-color: #EEF2F6;
        color: #2D3748;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Sidebar Soft UI */
    section[data-testid="stSidebar"] {
        background-color: #EEF2F6;
        border-right: 1px solid rgba(255, 255, 255, 0.7);
    }
    section[data-testid="stSidebar"] > div {
        background-color: #EEF2F6;
    }

    /* Neomorphic Headers */
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.98rem;
        color: #64748B;
        margin-bottom: 1.6rem;
        font-weight: 500;
    }

    /* Neomorphic Extruded Container Card */
    .neo-card {
        background: #EEF2F6;
        border-radius: 16px;
        box-shadow: 7px 7px 14px #d1d9e6, -7px -7px 14px #ffffff;
        padding: 20px 24px;
        margin-bottom: 18px;
        border: 1px solid rgba(255, 255, 255, 0.65);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .neo-card:hover {
        box-shadow: 9px 9px 18px #c8d2e0, -9px -9px 18px #ffffff;
    }

    /* Neomorphic Inset Metric Card */
    div[data-testid="stMetric"] {
        background: #EEF2F6;
        border-radius: 14px;
        box-shadow: 5px 5px 10px #d1d9e6, -5px -5px 10px #ffffff;
        padding: 12px 18px;
        border: 1px solid rgba(255, 255, 255, 0.7);
    }

    /* Neomorphic Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 14px;
        background-color: transparent;
        padding: 6px 0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #EEF2F6;
        border-radius: 12px;
        padding: 9px 22px;
        font-weight: 600;
        color: #64748B;
        box-shadow: 4px 4px 8px #d1d9e6, -4px -4px 8px #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.6) !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EEF2F6 !important;
        color: #2563EB !important;
        box-shadow: inset 3px 3px 6px #d1d9e6, inset -3px -3px 6px #ffffff !important;
        border: 1px solid rgba(209, 217, 230, 0.5) !important;
    }

    /* Soft UI Inset Inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        background-color: #EEF2F6 !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.7) !important;
        box-shadow: inset 3px 3px 6px #d1d9e6, inset -3px -3px 6px #ffffff !important;
        color: #1E293B !important;
    }

    /* Difficulty Badges - Soft UI Extruded Pills */
    .badge-neo {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        box-shadow: 2px 2px 5px #d1d9e6, -2px -2px 5px #ffffff;
    }
    .badge-beginner {
        background: #EEF2F6;
        color: #0D9488;
        border: 1px solid rgba(13, 148, 136, 0.3);
    }
    .badge-easy {
        background: #EEF2F6;
        color: #2563EB;
        border: 1px solid rgba(37, 99, 235, 0.3);
    }
    .badge-medium {
        background: #EEF2F6;
        color: #D97706;
        border: 1px solid rgba(217, 119, 6, 0.3);
    }
    .badge-hard {
        background: #EEF2F6;
        color: #DC2626;
        border: 1px solid rgba(220, 38, 38, 0.3);
    }
    .badge-challenge {
        background: #EEF2F6;
        color: #7C3AED;
        border: 1px solid rgba(124, 58, 237, 0.3);
    }

    /* Inset Score & Tag Indicators */
    .pill-inset {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        background: #EEF2F6;
        color: #475569;
        box-shadow: inset 2px 2px 4px #d1d9e6, inset -2px -2px 4px #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.4);
    }

    /* Soft UI Comparison Cards */
    .comp-box {
        background: #EEF2F6;
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 4px 4px 9px #d1d9e6, -4px -4px 9px #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.65);
    }
    .comp-box-kw { border-left: 4px solid #64748B; }
    .comp-box-sem { border-left: 4px solid #059669; }
    .comp-box-both { border-left: 4px solid #2563EB; }

    /* Expanders styling */
    div[data-testid="stExpander"] {
        background: #EEF2F6;
        border-radius: 12px;
        box-shadow: 3px 3px 6px #d1d9e6, -3px -3px 6px #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.5);
        margin-top: 6px;
    }
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
    
    df = attach_solution_stats(df, "data/solutions.csv")
    emb = np.load(emb_path)
    coords = np.load(coords_path) if os.path.exists(coords_path) else None
    searcher = Searcher(df, emb)
    return df, emb, coords, searcher

df, emb, coords, searcher = load_app_data()

# Sidebar
with st.sidebar:
    st.markdown("### **Deja Code**")
    st.caption("Competitive Programming Semantic Search & Recommendations")
    st.divider()
    if df is not None:
        st.metric("Problems Indexed", f"{len(df):,}")
        clusters_count = df["cluster"].nunique() if "cluster" in df.columns else 0
        st.metric("Concept Clusters", f"{clusters_count}")
        st.metric("Total Submissions Indexed", f"{df['total_submissions'].sum():,}")
    st.divider()
    st.markdown("Engineered with **Sentence-Transformers**, **scikit-learn**, **UMAP**, and **Streamlit**.")
    st.markdown("[GitHub Repository](https://github.com/visshalb2007/dejacode)")

st.markdown('<div class="main-header">Deja Code</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Algorithmic semantic search, topic clustering, and difficulty progression for CodeChef problems</div>', unsafe_allow_html=True)

if searcher is None:
    st.error("Index artifacts not found in out/. Please run python dejacode.py --data data/questions.csv --k 12 first.")
    st.stop()

tabs = st.tabs([
    "Search & Discovery",
    "Mode Comparison",
    "Cluster Manifold",
    "Recommendations"
])

# ----------------- TAB 1: Search & Discovery -----------------
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
            alpha = st.slider("Dense / Sparse Weight (alpha)", 0.0, 1.0, 0.65, 0.05, help="1.0 = Pure Semantic Dense, 0.0 = Pure Sparse Keyword")
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
        
        st.markdown(f"#### Top {len(top_k)} Matching Problems ({mode} mode)")
        
        for _, row in top_k.iterrows():
            diff = str(row.get("difficulty", "medium")).lower()
            badge_class = f"badge-{diff}" if diff in DIFF_ORDER else "badge-medium"
            codechef_link = f"https://www.codechef.com{row['link']}" if str(row.get("link", "")).startswith("/") else f"https://www.codechef.com/problems/{row.get('qcode', '')}"
            tags_str = ", ".join(row["tags"]) if isinstance(row["tags"], list) and row["tags"] else "None"
            
            with st.container():
                st.markdown(f"""
                <div class="neo-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #0F172A;">
                            <a href="{codechef_link}" target="_blank" style="text-decoration: none; color: #2563EB;">{row.get('title', 'Unknown')}</a>
                            <span style="font-size: 0.85rem; color: #64748B; font-weight: normal;">({row.get('qcode', 'N/A')})</span>
                        </h4>
                        <div>
                            <span class="badge-neo {badge_class}">{diff.upper()}</span>
                            <span class="pill-inset" style="margin-left: 8px;">Score: {row['score']:.3f}</span>
                        </div>
                    </div>
                    <div style="margin-top: 10px; font-size: 0.88rem; color: #475569;">
                        <strong>Tags:</strong> {tags_str} | 
                        <strong>Cluster:</strong> {row.get('cluster_label', 'N/A')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("Preview Problem Statement"):
                    st.write(row.get("text", "No statement text available."))

# ----------------- TAB 2: Mode Comparison -----------------
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
            st.markdown("#### Sparse Keyword (TF-IDF)")
            st.caption("Matches exact word tokens (often distracted by narrative words like Chef, city, game).")
            for rank, (_, row) in enumerate(kw_df.iterrows(), 1):
                is_shared = row["qcode"] in common
                box_class = "comp-box comp-box-both" if is_shared else "comp-box comp-box-kw"
                shared_tag = " <span class='pill-inset' style='color: #2563EB;'>In Both</span>" if is_shared else ""
                diff = str(row.get("difficulty", "medium")).lower()
                badge_class = f"badge-{diff}" if diff in DIFF_ORDER else "badge-medium"
                link = f"https://www.codechef.com{row['link']}" if str(row.get("link", "")).startswith("/") else f"https://www.codechef.com/problems/{row.get('qcode', '')}"
                
                st.markdown(f"""
                <div class="{box_class}">
                    <div style="font-weight: 600;">#{rank}. <a href="{link}" target="_blank" style="text-decoration: none; color: #1E293B;">{row['title']}</a> <span style="font-size: 0.85rem; color: #64748B;">({row['qcode']})</span>{shared_tag}</div>
                    <div style="font-size: 0.82rem; color: #64748B; margin-top: 6px;">
                        <span class="badge-neo {badge_class}">{diff.upper()}</span>
                        <span class="pill-inset" style="margin-left: 6px;">Score: {row['score']:.3f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with c_right:
            st.markdown("#### Dense Semantic (MiniLM-L6-v2)")
            st.caption("Matches conceptual problem meaning and algorithm abstraction across disparate stories.")
            for rank, (_, row) in enumerate(sem_df.iterrows(), 1):
                is_shared = row["qcode"] in common
                box_class = "comp-box comp-box-both" if is_shared else "comp-box comp-box-sem"
                shared_tag = " <span class='pill-inset' style='color: #2563EB;'>In Both</span>" if is_shared else " <span class='pill-inset' style='color: #059669;'>Semantic Match</span>"
                diff = str(row.get("difficulty", "medium")).lower()
                badge_class = f"badge-{diff}" if diff in DIFF_ORDER else "badge-medium"
                link = f"https://www.codechef.com{row['link']}" if str(row.get("link", "")).startswith("/") else f"https://www.codechef.com/problems/{row.get('qcode', '')}"

                st.markdown(f"""
                <div class="{box_class}">
                    <div style="font-weight: 600;">#{rank}. <a href="{link}" target="_blank" style="text-decoration: none; color: #1E293B;">{row['title']}</a> <span style="font-size: 0.85rem; color: #64748B;">({row['qcode']})</span>{shared_tag}</div>
                    <div style="font-size: 0.82rem; color: #64748B; margin-top: 6px;">
                        <span class="badge-neo {badge_class}">{diff.upper()}</span>
                        <span class="pill-inset" style="margin-left: 6px;">Score: {row['score']:.3f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ----------------- TAB 3: Cluster Manifold -----------------
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
            paper_bgcolor="#EEF2F6",
            plot_bgcolor="#EEF2F6",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=40, b=80),
            xaxis=dict(title="UMAP Component 1", showgrid=True, gridcolor="#E2E8F0"),
            yaxis=dict(title="UMAP Component 2", showgrid=True, gridcolor="#E2E8F0")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        summary_path = "out/cluster_summary.csv"
        if os.path.exists(summary_path):
            st.markdown("#### Cluster Profiles & Exemplar Medoids")
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
            
        hidden_path = "out/hidden_matches.csv"
        if os.path.exists(hidden_path):
            with st.expander("Discovered Hidden Matches (High Semantic Similarity, Low Lexical Overlap)"):
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
        st.info("2D coordinates not available. Run python dejacode.py --data data/questions.csv to generate projection.")

# ----------------- TAB 4: Recommendations -----------------
with tabs[3]:
    st.subheader("Adaptive Concept & Difficulty Progression")
    st.caption("Select problems you have solved to get concept-aligned recommendations calibrated to your target difficulty level.")

    @st.cache_data(show_spinner=False)
    def get_accepted_solution_map(solutions_path="data/solutions.csv"):
        if not os.path.exists(solutions_path): return {}
        try:
            sol = pd.read_csv(solutions_path, usecols=["QCode", "SolutionID", "Status", "Language"])
            acc = sol[sol["Status"].astype(str).str.lower() == "accepted"].drop_duplicates(subset=["QCode"])
            return dict(zip(acc["QCode"], zip(acc["SolutionID"], acc["Language"])))
        except Exception:
            return {}

    sol_map = get_accepted_solution_map()

    col_rec_in, col_rec_opt = st.columns([3, 1])

    with col_rec_in:
        problem_options = [f"{row['qcode']} | {row['title']} ({str(row.get('difficulty', 'medium')).capitalize()})" for _, row in df.iterrows()]
        default_selected = [problem_options[0], problem_options[1]] if len(problem_options) >= 2 else []
        
        preset = st.radio("Quick History Presets", ["None", "Beginner: FLOW017, CFRTEST", "Easy: SUMPAIR, CHEFLCM", "Medium: ANDTUPLE, DIVSUBS"], horizontal=True)
        if "FLOW017" in preset:
            pre_selected = [opt for opt in problem_options if "FLOW017" in opt or "CFRTEST" in opt]
        elif "SUMPAIR" in preset:
            pre_selected = [opt for opt in problem_options if "SUMPAIR" in opt or "CHEFLCM" in opt]
        elif "ANDTUPLE" in preset:
            pre_selected = [opt for opt in problem_options if "ANDTUPLE" in opt or "DIVSUBS" in opt]
        else:
            pre_selected = default_selected

        selected_problems = st.multiselect(
            "Select problems you have solved:",
            options=problem_options,
            default=pre_selected
        )

    with col_rec_opt:
        step_choice = st.radio(
            "Progression Step",
            [
                "+1 Tier (Level Up - Recommended)",
                "0 (Reinforce Same Tier)",
                "-1 Tier (Review Foundations)"
            ],
            index=0
        )
        step_val = 1 if "+1" in step_choice else (0 if "0" in step_choice else -1)
        k_recs = st.slider("Recommendations Count", 3, 10, 5)

    if selected_problems:
        solved_codes = [p.split(" | ")[0].strip() for p in selected_problems]
        recs_df = recommend_next(solved_codes, df, emb, k=k_recs, step=step_val)

        if not recs_df.empty:
            st.markdown(f"#### Recommended Next Problems ({recs_df.iloc[0]['difficulty'].upper()} tier)")

            for idx, (_, rec) in enumerate(recs_df.iterrows(), 1):
                diff = str(rec.get("difficulty", "medium")).lower()
                badge_class = f"badge-{diff}" if diff in DIFF_ORDER else "badge-medium"
                link = f"https://www.codechef.com{rec['link']}" if str(rec.get("link", "")).startswith("/") else f"https://www.codechef.com/problems/{rec.get('qcode', '')}"
                tags_str = ", ".join(rec["tags"]) if isinstance(rec["tags"], list) and rec["tags"] else "None"
                acc_rate = f"{rec.get('acceptance_rate', 0.0) * 100:.1f}%" if "acceptance_rate" in rec else "N/A"
                accepted_n = f"{int(rec.get('accepted_count', 0)):,}" if "accepted_count" in rec else "N/A"
                subs_n = f"{int(rec.get('total_submissions', 0)):,}" if "total_submissions" in rec else "N/A"

                with st.container():
                    st.markdown(f"""
                    <div class="neo-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4 style="margin: 0;">
                                #{idx}. <a href="{link}" target="_blank" style="text-decoration: none; color: #2563EB;">{rec.get('title', 'Unknown')}</a>
                                <span style="font-size: 0.85rem; color: #64748B; font-weight: normal;">({rec.get('qcode', 'N/A')})</span>
                            </h4>
                            <div>
                                <span class="badge-neo {badge_class}">{diff.upper()}</span>
                                <span class="pill-inset" style="margin-left: 8px;">Concept Sim: {rec.get('similarity', 0.0):.3f}</span>
                            </div>
                        </div>
                        <div style="margin-top: 10px; font-size: 0.88rem; color: #475569;">
                            <strong>Tags:</strong> {tags_str} |
                            <strong>Acceptance Rate:</strong> {acc_rate} ({accepted_n} / {subs_n} accepted)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander(f"Accepted Solution Code for {rec['qcode']}"):
                        sol_info = sol_map.get(rec["qcode"])
                        sol_id, lang = sol_info if sol_info else (None, None)
                        if sol_id:
                            code_snippet = get_code_snippet(sol_id)
                            if code_snippet:
                                st.caption(f"Accepted Submission: `{sol_id}` | Language: `{lang}`")
                                code_lang = "python" if "py" in str(lang).lower() else ("java" if "java" in str(lang).lower() else "cpp")
                                st.code(code_snippet, language=code_lang)
                            else:
                                st.info(f"Solution ID {sol_id} found in solutions.csv, but raw code file is not in program_codes/.")
                        else:
                            st.info(f"No accepted submission records found in solutions.csv for {rec['qcode']}.")
        else:
            st.warning("No candidate problems found matching the selected criteria.")
    else:
        st.info("Please select at least one solved problem above to generate recommendations.")
