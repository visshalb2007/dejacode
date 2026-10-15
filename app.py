import os, ast, numpy as np, pandas as pd, streamlit as st
from dejacode import Searcher

st.set_page_config(page_title="Déjà Code", layout="wide")
st.title("Déjà Code - Semantic Problem Search")

@st.cache_resource
def get_searcher():
    data_path = "out/problems_clustered.csv"
    emb_path = "out/emb.npy"
    if not (os.path.exists(data_path) and os.path.exists(emb_path)):
        return None
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
    return Searcher(df, np.load(emb_path))

s = get_searcher()
if s is None:
    st.warning("Artifacts not found in out/. Please run: `python dejacode.py --data data/questions.csv --k 10` first.")
else:
    q = st.text_area("Paste a problem statement or query", placeholder="e.g. Find shortest path in a weighted directed acyclic graph...")
    mode = st.radio("Mode", ["semantic", "hybrid", "keyword"], horizontal=True)
    k = st.slider("Results", 3, 20, 5)
    if q:
        r = s.search(q, k, mode)
        if "cluster_label" in s.df.columns:
            r["cluster"] = s.df.loc[r.index, "cluster_label"]
        st.dataframe(r, use_container_width=True)
    if os.path.exists("out/clusters.png"):
        st.image("out/clusters.png", caption="Problem clusters (PCA)")
