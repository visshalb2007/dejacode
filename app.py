import numpy as np, pandas as pd, streamlit as st
from dejacode import Searcher

st.title("Déjà Code - semantic problem search")

@st.cache_resource
def get():
    df = pd.read_csv("out/problems_clustered.csv")
    df["tags"] = df["tags"].map(eval)
    df["doc"] = df["title"].astype(str) + ". " + df["text"].astype(str)
    return Searcher(df, np.load("out/emb.npy"))

s = get()
q = st.text_area("Paste a problem statement")
mode = st.radio("Mode", ["semantic", "hybrid", "keyword"], horizontal=True)
k = st.slider("Results", 3, 20, 5)
if q:
    r = s.search(q, k, mode)
    r["cluster"] = s.df.loc[r.index, "cluster_label"]
    st.dataframe(r)
st.image("out/clusters.png", caption="Problem clusters")
