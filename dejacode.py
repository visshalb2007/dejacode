"""Deja Code core: preprocess -> embed -> search -> cluster -> evaluate."""
import re, ast, argparse, os
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

COLS = {  # candidate column names (lowercase) -> standard name
    "qcode": ["qcode", "code", "problem_code", "id"],
    "text": ["problem_statement", "statement", "description", "problem", "body", "text", "question"],
    "title": ["title", "name", "problem_name", "problem_title"],
    "tags": ["tags", "topics", "category", "categories", "algorithm_tags"],
    "difficulty": ["level", "difficulty", "rating", "difficulty_rating"],
    "link": ["link", "url", "problem_url"],
}

# ---------- 1. load + preprocess ----------
def load(path):
    if path.endswith(".json"): return pd.read_json(path)
    if path.endswith(".parquet"): return pd.read_parquet(path)
    return pd.read_csv(path)

def clean(t):
    t = re.sub(r"<[^>]+>", " ", str(t))             # html
    t = re.sub(r"(?i)all submissions for this problem are available\.?", " ", t)
    t = re.sub(r"(?i)read problems? statements? in [^.]*?(?:as well|here)\s*\.?|read problems? statements? in [^.]*\.", " ", t)
    t = re.sub(r"\$+([^$]*)\$+", r" \1 ", t)         # latex $..$
    t = re.sub(r"http\S+", " ", t)                   # urls
    t = re.sub(r"(?i)\b(input|output|constraints|sample (input|output)|example)s?\b\s*:?", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def parse_tags(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return []
    if isinstance(v, (list, tuple)): return [str(x).strip().lower() for x in v]
    s = str(v).strip()
    try:
        r = ast.literal_eval(s)
        if isinstance(r, (list, tuple)): return [str(x).strip().lower() for x in r]
    except Exception: pass
    return [x.strip().lower() for x in re.split(r"[,;|]", s) if x.strip()]

SYN = {"dynamic-prog": "dp", "dynamic-programming": "dp", "maths": "math", "mathematics": "math",
       "simple-math": "math", "ad-hoc": "adhoc", "strings": "string"}
NOISE = {"easy", "medium", "hard", "easy-medium", "medium-hard", "cakewalk", "challenge",
         "beginner", "simple", "basic", "basics", "admin", "admin2"}

def topic_tags(tags, who):
    """Keep real topic tags only: drop author handles, contest codes (cook64, jan15), difficulty words."""
    return [SYN.get(t, t) for t in tags
            if t not in NOISE and t not in who and not re.search(r"\d{2}$", t)]

def preprocess(df):
    low = {c.lower(): c for c in df.columns}
    out = pd.DataFrame()
    for std, cands in COLS.items():
        col = next((low[c] for c in cands if c in low), None)
        out[std] = df[col] if col else (None if std != "text" else "")
    if out["text"].eq("").all(): raise ValueError(f"No text column found in {list(df.columns)}")
    out["title"] = out["title"].fillna("untitled").astype(str)
    out["text"] = out["text"].fillna("").map(clean)
    who = set()
    for c in ("author", "tester"):
        if c in low:
            for v in df[low[c]].dropna().astype(str):
                who |= {x.strip().lower() for x in v.split(",")}
    out["tags"] = out["tags"].map(lambda v: topic_tags(parse_tags(v), who))
    out = out[~out["text"].str.contains("non-commercial competitive programming|function formatTime", regex=True)]  # scraped page junk
    out = out[out["text"].str.len() > 30].drop_duplicates("text").reset_index(drop=True)
    out["doc"] = out["title"] + ". " + out["text"]
    return out

# ---------- Solution & Code helpers ----------
def get_solution_stats(solutions_path="data/solutions.csv"):
    """Return DataFrame indexed by QCode with total_submissions, accepted_count, acceptance_rate."""
    if not os.path.exists(solutions_path):
        return pd.DataFrame(columns=["total_submissions", "accepted_count", "acceptance_rate"])
    df = pd.read_csv(solutions_path, usecols=["QCode", "Status"])
    total = df.groupby("QCode").size().rename("total_submissions")
    acc = df[df["Status"] == "accepted"].groupby("QCode").size().rename("accepted_count")
    stats = pd.concat([total, acc], axis=1).fillna(0).astype({"total_submissions": int, "accepted_count": int})
    stats["acceptance_rate"] = (stats["accepted_count"] / stats["total_submissions"].clip(lower=1)).round(3)
    return stats

def get_code_snippet(solution_id, codes_dir="data/program_codes", max_chars=1200):
    """Retrieve solution code snippet by SolutionID without loading all code into memory."""
    if not os.path.exists(codes_dir): return None
    for fn in ["first.csv", "second.csv", "third.csv"]:
        path = os.path.join(codes_dir, fn)
        if not os.path.exists(path): continue
        try:
            for chunk in pd.read_csv(path, chunksize=15000):
                m = chunk[chunk["SolutionID"] == solution_id]
                if not m.empty:
                    code = str(m.iloc[0]["Solutions"]).strip()
                    return code[:max_chars]
        except Exception:
            continue
    return None

# ---------- 2. embeddings ----------
def embed(texts, model="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model)
    return m.encode(list(texts), batch_size=64, show_progress_bar=True, normalize_embeddings=True)

# ---------- 3. search (semantic / keyword / hybrid) ----------
class Searcher:
    def __init__(self, df, emb, model="all-MiniLM-L6-v2"):
        self.df, self.emb, self.model_name, self._m = df, emb, model, None
        self.tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=50000)
        self.tf = self.tfidf.fit_transform(df["doc"])

    def _q_emb(self, q):
        if self._m is None:
            from sentence_transformers import SentenceTransformer
            self._m = SentenceTransformer(self.model_name)
        return self._m.encode([q], normalize_embeddings=True)[0]

    def scores(self, q, mode="semantic", alpha=0.7, qi=None):
        if qi is not None:
            sem = self.emb @ self.emb[qi]
            kw = (self.tf @ self.tf[qi].T).toarray().ravel()
        else:
            sem = self.emb @ self._q_emb(q)
            kw = (self.tf @ self.tfidf.transform([q]).T).toarray().ravel()
        if mode == "semantic": return sem.copy()
        if mode == "keyword": return kw.copy()
        z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
        return alpha * z(sem) + (1 - alpha) * z(kw)

    def search(self, q, k=5, mode="semantic", alpha=0.7, exclude=None, qi=None):
        s = self.scores(q, mode, alpha, qi=qi)
        if exclude is not None: s[exclude] = -1e9
        idx = np.argsort(-s)[:k]
        cols = [c for c in ["qcode", "title", "difficulty", "tags", "link"] if c in self.df.columns]
        r = self.df.iloc[idx][cols].copy()
        r["score"] = s[idx]
        return r

# ---------- 4. clustering + labels ----------
def cluster_hdbscan(emb, min_cluster_size=10, min_samples=3):
    from sklearn.cluster import HDBSCAN
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, copy=True)
    labels = hdb.fit_predict(emb)
    mask = labels != -1
    n_c = len(set(labels) - {-1})
    sil = silhouette_score(emb[mask], labels[mask]) if (n_c > 1 and mask.sum() > n_c) else 0.0
    return labels, sil

def cluster(emb, method="kmeans", k=12, min_cluster_size=10, min_samples=3, seed=0):
    if method == "hdbscan":
        return cluster_hdbscan(emb, min_cluster_size=min_cluster_size, min_samples=min_samples)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(emb)
    return km.labels_, silhouette_score(emb, km.labels_, sample_size=min(5000, len(emb)), random_state=seed)

def extract_cluster_topic_tags(df, labels, topn=3):
    """Extract dominant topic tags by frequency within each cluster."""
    from collections import Counter
    out = {}
    for c in sorted(set(labels)):
        mask = np.array(labels) == c
        sub_tags = []
        for t in df.loc[mask, "tags"]:
            if isinstance(t, (list, tuple)):
                sub_tags.extend(t)
            elif isinstance(t, str) and t.startswith("["):
                try: sub_tags.extend(ast.literal_eval(t))
                except Exception: pass
        counts = Counter(sub_tags).most_common(topn)
        out[c] = [tag for tag, _ in counts]
    return out

CONTEST_STOP_WORDS = {
    'test', 'case', 'cases', 'line', 'chef', 'number', 'numbers', 'integers', 'integer',
    'input', 'output', 'given', 'print', 'find', 'problem', 'task', 'following', 'example',
    'sample', 'time', 'constraints', 'subtask', 'contains'
}

def find_cluster_exemplars(df, emb, labels, topn=3):
    """Find the topn problems closest to the centroid of each cluster."""
    out = {}
    for c in sorted(set(labels)):
        if c == -1:
            out[c] = []
            continue
        mask = np.array(labels) == c
        sub_indices = np.where(mask)[0]
        if len(sub_indices) == 0:
            out[c] = []
            continue
        sub_emb = emb[sub_indices]
        centroid = sub_emb.mean(axis=0, keepdims=True)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        sims = (sub_emb @ centroid.T).ravel()
        top_ranks = np.argsort(-sims)[:min(topn, len(sub_indices))]
        exemplars = []
        for r in top_ranks:
            orig_idx = sub_indices[r]
            row = df.iloc[orig_idx]
            qcode = row.get("qcode", f"P{orig_idx}")
            title = row.get("title", "Untitled")
            exemplars.append(f"{qcode} ({title})")
        out[c] = exemplars
    return out

def label_clusters(df, labels, topn=5):
    """Generate clean readable cluster labels combining top topic tags and distinctive TF-IDF terms."""
    tv = TfidfVectorizer(stop_words="english", max_features=20000)
    X = tv.fit_transform(df["doc"]); terms = np.array(tv.get_feature_names_out())
    topic_tags_map = extract_cluster_topic_tags(df, labels, topn=3)
    out = {}
    for c in sorted(set(labels)):
        if c == -1:
            out[c] = "Noise / Outliers"
            continue
        mask = np.array(labels) == c
        m = np.asarray(X[mask].mean(axis=0)).ravel()
        top_terms = [t for t in terms[np.argsort(-m)] if t not in CONTEST_STOP_WORDS][:topn]
        tags = topic_tags_map.get(c, [])
        tag_prefix = f"[{', '.join(tags)}] " if tags else ""
        out[c] = f"{tag_prefix}{', '.join(top_terms)}"
    return out

def export_cluster_summary(df, emb, labels, names, out_path="out/cluster_summary.csv"):
    """Export readable cluster profiles, topic tags, TF-IDF terms, and 3 exemplar problems."""
    topic_tags_map = extract_cluster_topic_tags(df, labels, topn=5)
    exemplars_map = find_cluster_exemplars(df, emb, labels, topn=3)
    tv = TfidfVectorizer(stop_words="english", max_features=20000)
    X = tv.fit_transform(df["doc"]); terms = np.array(tv.get_feature_names_out())
    rows = []
    for c in sorted(set(labels)):
        mask = np.array(labels) == c
        size = int(mask.sum())
        m = np.asarray(X[mask].mean(axis=0)).ravel()
        top_terms = [t for t in terms[np.argsort(-m)] if t not in CONTEST_STOP_WORDS][:5]
        exs = exemplars_map.get(c, [])
        rows.append({
            "cluster_id": c,
            "cluster_label": names.get(c, ""),
            "top_topic_tags": ", ".join(topic_tags_map.get(c, [])),
            "top_tfidf_terms": ", ".join(top_terms),
            "size": size,
            "exemplar_1": exs[0] if len(exs) > 0 else "",
            "exemplar_2": exs[1] if len(exs) > 1 else "",
            "exemplar_3": exs[2] if len(exs) > 2 else "",
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_path, index=False)
    return summary_df

# ---------- 5. visualization ----------
def project_2d(emb, method="umap", seed=0):
    """Project embeddings to 2D using UMAP (nonlinear) or PCA (linear fallback)."""
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed)
            return reducer.fit_transform(emb)
        except Exception as e:
            print(f"UMAP projection failed ({e}), falling back to PCA.")
    return PCA(2, random_state=seed).fit_transform(emb)

def plot_clusters(emb, labels, names, path="out/clusters.png", xy=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if xy is None:
        xy = PCA(2, random_state=0).fit_transform(emb)
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab20", s=6)
    for c, n in names.items():
        if c == -1: continue
        mask = labels == c
        if mask.any():
            p = xy[mask].mean(axis=0); plt.text(p[0], p[1], str(c), weight="bold")
    plt.title("Problem clusters"); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_interactive_clusters(df, xy, labels, names, path="out/clusters.html"):
    """Generate an interactive Plotly HTML scatter with cluster colors and rich hover info."""
    import plotly.graph_objects as go
    hover_texts = []
    cluster_names = [f"Cluster {c}: {names.get(c, 'Outliers/Noise' if c == -1 else 'Cluster ' + str(c))}" for c in labels]
    for i, (_, row) in enumerate(df.iterrows()):
        qcode = row.get("qcode", "")
        title = row.get("title", "")
        diff = row.get("difficulty", "unknown")
        tags_str = ", ".join(row.get("tags", [])) if isinstance(row.get("tags"), list) else str(row.get("tags", ""))
        snippet = str(row.get("text", ""))[:140].replace("<", "&lt;").replace(">", "&gt;")
        hover_texts.append(
            f"<b>{title}</b> ({qcode})<br>"
            f"<b>Difficulty:</b> {diff}<br>"
            f"<b>Cluster:</b> {cluster_names[i]}<br>"
            f"<b>Tags:</b> {tags_str}<br>"
            f"<i>{snippet}...</i>"
        )
    fig = go.Figure()
    unique_clusters = sorted(set(labels))
    for c in unique_clusters:
        mask = np.array(labels) == c
        c_label = names.get(c, "Outliers/Noise") if c != -1 else "Noise/Outliers"
        c_name = f"Cluster {c}: {c_label[:30]}"
        fig.add_trace(go.Scatter(
            x=xy[mask, 0],
            y=xy[mask, 1],
            mode="markers",
            name=c_name,
            marker=dict(size=6, opacity=0.75),
            text=[hover_texts[idx] for idx, m in enumerate(mask) if m],
            hoverinfo="text"
        ))
    fig.update_layout(
        title="Déjà Code - Interactive Problem Clusters (UMAP 2D)",
        xaxis=dict(title="Dim 1", showgrid=True, zeroline=False),
        yaxis=dict(title="Dim 2", showgrid=True, zeroline=False),
        hovermode="closest",
        legend=dict(itemsizing="constant", orientation="v"),
        width=1100,
        height=750,
        template="plotly_white"
    )
    fig.write_html(path, include_plotlyjs="cdn")
    return fig

def explain_hidden_match(r1, r2):
    """Generate an explanation of why two problems match algorithmically despite different narratives."""
    t1, t2 = str(r1["text"]).lower(), str(r2["text"]).lower()
    t1_title, t2_title = r1.get("title", ""), r2.get("title", "")
    concepts = [
        ("shortest-path & graph traversal", ["shortest", "path", "dist", "graph", "vertices", "edges", "travel", "cities"]),
        ("2D grid & matrix dynamic programming", ["grid", "cell", "cells", "matrix", "walk", "moves", "triangle"]),
        ("substring & string matching", ["string", "strings", "substring", "characters", "prefix", "suffix", "palindrome"]),
        ("combinatorics & modular arithmetic", ["permutation", "combinations", "modulo", "ways", "permutations", "digits"]),
        ("turn-based game theory (Nim/Sprague-Grundy)", ["game", "player", "turn", "moves", "win", "optimal"]),
        ("subarray optimization & dynamic programming", ["array", "subarray", "contiguous", "sum", "maximum", "minimum"]),
        ("binary search & monotonic optimization", ["binary search", "sorted", "monotonic", "elements", "order", "search"]),
        ("bipartite graph & pairs matching", ["pair", "pairs", "matching", "friends", "partition", "graph"])
    ]
    detected = []
    for cname, words in concepts:
        s1 = sum(1 for w in words if w in t1)
        s2 = sum(1 for w in words if w in t2)
        if s1 >= 2 and s2 >= 2:
            detected.append(cname)
    core = detected[0] if detected else "structural algorithmic pattern"
    return f"Shared algorithmic core: {core}. Divergent narratives: '{t1_title}' vs '{t2_title}'."

# ---------- 5b. hidden matches (same algorithm, different story) ----------
def find_hidden_matches(df, emb, min_semantic=0.62, max_keyword=0.12, top_n=30):
    """Find problem pairs with high semantic embedding similarity but low TF-IDF keyword overlap."""
    tv = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=30000)
    X = tv.fit_transform(df["doc"])
    sparse_sim = (X @ X.T).toarray()
    dense_sim = emb @ emb.T
    
    i_idx, j_idx = np.triu_indices(len(df), k=1)
    d_vals = dense_sim[i_idx, j_idx]
    s_vals = sparse_sim[i_idx, j_idx]
    
    mask = (d_vals >= min_semantic) & (s_vals <= max_keyword)
    cand_i, cand_j = i_idx[mask], j_idx[mask]
    div_vals = d_vals[mask] - s_vals[mask]
    
    top_ranks = np.argsort(-div_vals)[:top_n]
    
    pairs = []
    for r in top_ranks:
        i, j = cand_i[r], cand_j[r]
        row_i, row_j = df.iloc[i], df.iloc[j]
        pairs.append({
            "prob1_code": row_i.get("qcode", f"P{i}"),
            "prob1_title": row_i.get("title", "Untitled"),
            "prob2_code": row_j.get("qcode", f"P{j}"),
            "prob2_title": row_j.get("title", "Untitled"),
            "semantic_sim": round(float(d_vals[mask][r]), 3),
            "keyword_sim": round(float(s_vals[mask][r]), 3),
            "divergence": round(float(div_vals[r]), 3),
            "cluster_1": row_i.get("cluster", -1),
            "cluster_2": row_j.get("cluster", -1),
            "explanation": explain_hidden_match(row_i, row_j)
        })
    return pd.DataFrame(pairs)

# ---------- 6. evaluation (relevance = shares >=1 tag) ----------
def evaluate(s, n_queries=200, k=10, seed=0):
    df = s.df; ok = np.where(df["tags"].map(len) > 0)[0]
    if len(ok) == 0: return None, []
    qs = np.random.default_rng(seed).choice(ok, min(n_queries, len(ok)), replace=False)
    res, fails = {}, []
    for mode in ["keyword", "semantic", "hybrid"]:
        P, MRR, NDCG = [], [], []
        for qi in qs:
            sc = s.scores(df["doc"][qi], mode, qi=qi); sc[qi] = -1e9
            top = np.argsort(-sc)[:k]; qt = set(df["tags"][qi])
            rel = np.array([len(qt & set(df["tags"][j])) > 0 for j in top], float)
            P.append(rel.mean())
            MRR.append(1 / (np.argmax(rel) + 1) if rel.any() else 0)
            NDCG.append((rel / np.log2(np.arange(2, k + 2))).sum() / max(1e-9, (np.sort(rel)[::-1] / np.log2(np.arange(2, k + 2))).sum()) if rel.any() else 0)
            if mode == "semantic" and rel[0] == 0:  # top-1 is a false positive
                fails.append((df["title"][qi], df["title"][top[0]], round(float(sc[top[0]]), 3), sorted(qt), sorted(df["tags"][top[0]])))
        res[mode] = {f"P@{k}": np.mean(P), "MRR": np.mean(MRR), f"nDCG@{k}": np.mean(NDCG)}
    return pd.DataFrame(res).T, fails

def evaluate_curated(s, eval_path="data/eval_set.csv", k=10):
    """Evaluate against curated benchmark with known-similar problem sets (P@5, MRR, nDCG@k)."""
    if not os.path.exists(eval_path): return None
    eval_df = pd.read_csv(eval_path)
    df = s.df
    qcode_to_idx = {str(q): i for i, q in enumerate(df["qcode"])} if "qcode" in df.columns else {}
    valid_rows = [row for _, row in eval_df.iterrows() if str(row["query_qcode"]) in qcode_to_idx]
    if not valid_rows: return None
    res = {}
    for mode in ["keyword", "semantic", "hybrid"]:
        P5, MRR, NDCG = [], [], []
        for row in valid_rows:
            qi = qcode_to_idx[str(row["query_qcode"])]
            targets = {x.strip() for x in str(row["similar_qcodes"]).split(",") if x.strip()}
            sc = s.scores(df["doc"][qi], mode, qi=qi)
            sc[qi] = -1e9
            top = np.argsort(-sc)[:k]
            top_qcodes = [str(df.iloc[j]["qcode"]) for j in top] if "qcode" in df.columns else []
            rel = np.array([1.0 if q in targets else 0.0 for q in top_qcodes], float)
            P5.append(rel[:5].mean() if len(rel) >= 5 else rel.mean())
            MRR.append(1.0 / (np.argmax(rel) + 1) if rel.any() else 0.0)
            discounts = np.log2(np.arange(2, len(rel) + 2))
            dcg = (rel / discounts).sum()
            n_rel = min(len(targets), k)
            idcg = (1.0 / np.log2(np.arange(2, n_rel + 2))).sum() if n_rel > 0 else 1.0
            NDCG.append(dcg / max(1e-9, idcg))
        res[mode] = {"P@5": np.mean(P5), "MRR": np.mean(MRR), f"nDCG@{k}": np.mean(NDCG)}
    return pd.DataFrame(res).T

def tune_hybrid_alpha(s, eval_path="data/eval_set.csv", alphas=None):
    """Grid sweep alpha in hybrid search to find the optimal balance between keyword and semantic scores."""
    if not os.path.exists(eval_path): return None
    eval_df = pd.read_csv(eval_path)
    df = s.df
    qcode_to_idx = {str(q): i for i, q in enumerate(df["qcode"])} if "qcode" in df.columns else {}
    valid_rows = [row for _, row in eval_df.iterrows() if str(row["query_qcode"]) in qcode_to_idx]
    if not valid_rows: return None
    if alphas is None:
        alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    records = []
    for alpha in alphas:
        P5, MRR, NDCG = [], [], []
        for row in valid_rows:
            qi = qcode_to_idx[str(row["query_qcode"])]
            targets = {x.strip() for x in str(row["similar_qcodes"]).split(",") if x.strip()}
            mode = "keyword" if alpha == 0.0 else ("semantic" if alpha == 1.0 else "hybrid")
            sc = s.scores(df["doc"][qi], mode=mode, alpha=alpha, qi=qi)
            sc[qi] = -1e9
            top = np.argsort(-sc)[:10]
            top_qcodes = [str(df.iloc[j]["qcode"]) for j in top] if "qcode" in df.columns else []
            rel = np.array([1.0 if q in targets else 0.0 for q in top_qcodes], float)
            P5.append(rel[:5].mean() if len(rel) >= 5 else rel.mean())
            MRR.append(1.0 / (np.argmax(rel) + 1) if rel.any() else 0.0)
            discounts = np.log2(np.arange(2, len(rel) + 2))
            dcg = (rel / discounts).sum()
            n_rel = min(len(targets), 10)
            idcg = (1.0 / np.log2(np.arange(2, n_rel + 2))).sum() if n_rel > 0 else 1.0
            NDCG.append(dcg / max(1e-9, idcg))
        records.append({
            "alpha": alpha,
            "P@5": round(float(np.mean(P5)), 3),
            "MRR": round(float(np.mean(MRR)), 3),
            "nDCG@10": round(float(np.mean(NDCG)), 3)
        })
    return pd.DataFrame(records)

def compare_models(df, models=None, eval_path="data/eval_set.csv", out_dir="out"):
    """Evaluate and compare dense embedding models side-by-side on the curated benchmark."""
    if models is None:
        models = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
    results = []
    for model_name in models:
        slug = model_name.split("/")[-1]
        emb_file = os.path.join(out_dir, f"emb_{slug}.npy")
        if os.path.exists(emb_file):
            emb_m = np.load(emb_file)
        elif model_name == "all-MiniLM-L6-v2" and os.path.exists(os.path.join(out_dir, "emb.npy")):
            emb_m = np.load(os.path.join(out_dir, "emb.npy"))
        else:
            emb_m = embed(df["doc"], model=model_name)
            np.save(emb_file, emb_m)
        s_m = Searcher(df, emb_m, model=model_name)
        cur_m = evaluate_curated(s_m, eval_path, k=10)
        if cur_m is not None:
            for mode, row in cur_m.iterrows():
                results.append({
                    "model": slug,
                    "mode": mode,
                    "P@5": row["P@5"],
                    "MRR": row["MRR"],
                    "nDCG@10": row["nDCG@10"]
                })
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(out_dir, "model_tuning_results.csv"), index=False)
    return res_df

def categorize_failures(fails_df, df):
    """Categorize false match failures into 4 distinct failure causes."""
    title_to_text = dict(zip(df["title"], df["text"]))
    cats = []
    story_words = {"chef", "sereja", "elephant", "little elephant", "ciel", "johny", "alex", "friend", "friends", "game", "shop", "shopping", "coins"}
    syn_pairs = [{"math", "number-theory"}, {"graph", "tree"}, {"graph", "bfs"}, {"string", "palindrome"}, {"dp", "memoization"}]
    for _, row in fails_df.iterrows():
        q_title = str(row["query"]); m_title = str(row["wrong_match"])
        q_tags = row["q_tags"] if isinstance(row["q_tags"], (list, tuple)) else (ast.literal_eval(str(row["q_tags"])) if str(row["q_tags"]).startswith("[") else [])
        m_tags = row["m_tags"] if isinstance(row["m_tags"], (list, tuple)) else (ast.literal_eval(str(row["m_tags"])) if str(row["m_tags"]).startswith("[") else [])
        if len(m_tags) == 0:
            cats.append("Noisy Tags (Missing in Candidate)")
            continue
        tag_overlap_miss = False
        for sp in syn_pairs:
            if (set(q_tags) & sp) and (set(m_tags) & sp):
                cats.append("Noisy Tags (Synonym Incomplete)")
                tag_overlap_miss = True
                break
        if tag_overlap_miss: continue
        q_text = title_to_text.get(q_title, ""); m_text = title_to_text.get(m_title, "")
        if len(str(q_text)) < 180 or len(str(m_text)) < 180:
            cats.append("Short Ambiguous Statement")
            continue
        words_q = set(q_title.lower().split()); words_m = set(m_title.lower().split())
        if (words_q & words_m) & story_words:
            cats.append("Shared Story Words")
            continue
        if "test cases" in str(q_text).lower() and "test cases" in str(m_text).lower():
            cats.append("Shared Input Format")
            continue
        cats.append("Shared Story Words")
    fails_df["failure_cause"] = cats
    return fails_df

def export_failure_summary(fails_df, out_path="out/failure_summary.json"):
    """Export aggregated failure statistics and taxonomy distribution."""
    import json
    counts = fails_df["failure_cause"].value_counts().to_dict()
    total = len(fails_df)
    percentages = {k: round((v / total) * 100, 1) for k, v in counts.items()} if total > 0 else {}
    summary = {
        "total_failures": total,
        "counts": counts,
        "percentages": percentages,
        "top_examples": fails_df.groupby("failure_cause").head(2)[["query", "wrong_match", "score", "failure_cause"]].to_dict(orient="records")
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary

# ---------- CLI: build everything ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--model", default="all-MiniLM-L6-v2"); ap.add_argument("--out", default="out")
    ap.add_argument("--cluster-method", default="kmeans", choices=["kmeans", "hdbscan"])
    ap.add_argument("--min-cluster-size", type=int, default=10)
    ap.add_argument("--min-samples", type=int, default=3)
    ap.add_argument("--dim-red", default="umap", choices=["umap", "pca"])
    ap.add_argument("--force-embed", action="store_true", help="Force recomputation of embeddings")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    df = preprocess(load(a.data)); print("problems after cleaning:", len(df))
    emb_path = f"{a.out}/emb.npy"
    if os.path.exists(emb_path) and not a.force_embed:
        emb = np.load(emb_path)
        if len(emb) != len(df):
            emb = embed(df["doc"], a.model); np.save(emb_path, emb)
    else:
        emb = embed(df["doc"], a.model); np.save(emb_path, emb)
    labels, sil = cluster(emb, method=a.cluster_method, k=a.k, min_cluster_size=a.min_cluster_size, min_samples=a.min_samples)
    names = label_clusters(df, labels)
    df["cluster"] = labels; df["cluster_label"] = df["cluster"].map(names)
    df.drop(columns="doc").to_csv(f"{a.out}/problems_clustered.csv", index=False)
    summary_df = export_cluster_summary(df, emb, labels, names, f"{a.out}/cluster_summary.csv")
    print(f"Saved cluster summary to {a.out}/cluster_summary.csv")
    hidden_df = find_hidden_matches(df, emb, min_semantic=0.62, max_keyword=0.12, top_n=30)
    hidden_df.to_csv(f"{a.out}/hidden_matches.csv", index=False)
    print(f"Saved {len(hidden_df)} hidden matches to {a.out}/hidden_matches.csv")
    
    # 2D projection
    coords_path = f"{a.out}/coords_2d.npy"
    if os.path.exists(coords_path) and not a.force_embed:
        xy = np.load(coords_path)
    else:
        xy = project_2d(emb, method=a.dim_red)
        np.save(coords_path, xy)
        
    plot_clusters(emb, labels, names, f"{a.out}/clusters.png", xy=xy)
    plot_interactive_clusters(df, xy, labels, names, f"{a.out}/clusters.html")
    print(f"Clustering ({a.cluster_method}): silhouette={sil:.3f}")
    [print(f"Cluster {c}: {n}") for c, n in names.items()]
    searcher = Searcher(df, emb, a.model)
    metrics, fails = evaluate(searcher)
    if metrics is not None:
        print(metrics.round(3)); metrics.to_csv(f"{a.out}/metrics.csv")
        fails_df = pd.DataFrame(fails, columns=["query", "wrong_match", "score", "q_tags", "m_tags"])
        fails_df = categorize_failures(fails_df, df)
        fails_df.to_csv(f"{a.out}/failures.csv", index=False)
        fail_stats = export_failure_summary(fails_df, f"{a.out}/failure_summary.json")
        print("\nFailure Categories Breakdown:")
        for k, v in fail_stats.get("percentages", {}).items():
            print(f"  - {k}: {v}% ({fail_stats['counts'][k]})")
    curated_metrics = evaluate_curated(searcher, "data/eval_set.csv", k=10)
    if curated_metrics is not None:
        print("\n=== Curated Benchmark Evaluation (Hand-crafted Ground Truth) ===")
        print(curated_metrics.round(3)); curated_metrics.to_csv(f"{a.out}/eval_benchmark_metrics.csv")
