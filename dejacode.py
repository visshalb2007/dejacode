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

def label_clusters(df, labels, topn=5):
    tv = TfidfVectorizer(stop_words="english", max_features=20000)
    X = tv.fit_transform(df["doc"]); terms = np.array(tv.get_feature_names_out())
    out = {}
    for c in sorted(set(labels)):
        m = np.asarray(X[labels == c].mean(axis=0)).ravel()
        out[c] = ", ".join(terms[np.argsort(-m)[:topn]])
    return out

# ---------- 5. visualization ----------
def plot_clusters(emb, labels, names, path="out/clusters.png"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    xy = PCA(2, random_state=0).fit_transform(emb)
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab20", s=6)
    for c, n in names.items():
        p = xy[labels == c].mean(axis=0); plt.text(p[0], p[1], str(c), weight="bold")
    plt.title("Problem clusters (PCA)"); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

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

# ---------- CLI: build everything ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--model", default="all-MiniLM-L6-v2"); ap.add_argument("--out", default="out")
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
    labels, sil = cluster(emb, a.k); names = label_clusters(df, labels)
    df["cluster"] = labels; df["cluster_label"] = df["cluster"].map(names)
    df.drop(columns="doc").to_csv(f"{a.out}/problems_clustered.csv", index=False)
    plot_clusters(emb, labels, names, f"{a.out}/clusters.png")
    print(f"silhouette={sil:.3f}"); [print(c, n) for c, n in names.items()]
    searcher = Searcher(df, emb, a.model)
    metrics, fails = evaluate(searcher)
    if metrics is not None:
        print("\n=== Tag-Overlap Evaluation (Noisy Proxy) ===")
        print(metrics.round(3)); metrics.to_csv(f"{a.out}/metrics.csv")
        pd.DataFrame(fails, columns=["query", "wrong_match", "score", "q_tags", "m_tags"]).to_csv(f"{a.out}/failures.csv", index=False)
    curated_metrics = evaluate_curated(searcher, "data/eval_set.csv", k=10)
    if curated_metrics is not None:
        print("\n=== Curated Benchmark Evaluation (Hand-crafted Ground Truth) ===")
        print(curated_metrics.round(3)); curated_metrics.to_csv(f"{a.out}/eval_benchmark_metrics.csv")
