<div align="center">

<img width="1561" height="381" alt="image" src="https://github.com/user-attachments/assets/6eeb978b-1778-4033-86ee-c20784604600" />

### Dense Semantic Information Retrieval & Unsupervised Topic Discovery for Competitive Programming

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-Sentence--BERT-yellow.svg?logo=huggingface&logoColor=white)](https://huggingface.co/sentence-transformers)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

*A production-grade NLP and neural information retrieval engine designed to pierce through narrative story-dressing in algorithmic problem statements to discover isomorphic problems, thematic concept clusters, and optimal learning trajectories.*

</div>

---

## Executive Summary

In competitive programming ecosystems (CodeChef, Codeforces, LeetCode), problem authors deliberately shroud algorithmic specifications under complex narrative themes (e.g., *Chefs baking pies*, *knights traversing enchanted forests*, or *coin transactions in fictional economies*). 

Traditional information retrieval systems based on lexical frequencies (BM25, TF-IDF) suffer from catastrophic failure modes in this domain:
- **High False Negatives (Algorithmic Isomorphism):** Two mathematically identical problems (e.g., Finding the Minimum Spanning Tree of a weighted graph) share zero vocabulary tokens if one describes cities connected by roads and the other describes islands connected by shipping lanes.
- **High False Positives (Entity Overfitting):** Two completely unrelated problems (one trivial greedy sorting, the other NP-hard dynamic programming) score high lexical similarity simply because both mention character names like *"Chef"* or transactional words like *"store"* and *"coins"*.

**Déjà Code** implements a dense neural embedding retrieval pipeline, nonlinear manifold learning, and an adaptive difficulty ladder built on a curated dataset of **1,429 CodeChef problems** and **637,763 code submissions**.

---

## System Architecture

```
                          ┌───────────────────────────┐
                          │   Raw CodeChef Dataset    │
                          │   (1,429 Problems,        │
                          │   637k Code Submissions)  │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │       Noise-Filtering & Normalization Engine        │
             │  - Strip LaTeX, MathJax, HTML, URL artifacts         │
             │  - Filter competitive programming I/O boilerplate    │
             │  - Canonicalize algorithmic topic tags & synonyms   │
             └──────────┬───────────────────────────────┬──────────┘
                        │                               │
                        ▼                               ▼
       ┌─────────────────────────────────┐   ┌────────────────────────────────┐
       │   Dense Neural Encoder (384-d)  │   │  Sparse Lexical Index (50k)    │
       │   Sentence-BERT (MiniLM-L6-v2)  │   │  Sublinear TF-IDF + Custom CP  │
       │   Unit L2 Sphere Projection     │   │  Stopwords Filtering           │
       └────────────────┬────────────────┘   └────────────────┬───────────────┘
                        │                                     │
                        │                                     │
                        ▼                                     ▼
        ┌─────────────────────────────────────────────────────────────┐
        │            Hybrid Dynamic Retrieval Engine                  │
        │     Score = α · z(DenseCosine) + (1-α) · z(SparseTFIDF)     │
        │             Calibrated Optimal α = 0.65                     │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌───────────────────────────────┐             ┌──────────────────────────────────────┐
│  Manifold Learning & Clusters │             │     Curated Evaluation & Recs        │
│  - HDBSCAN / K-Means (k=12)   │             │  - 45-Query Hand-Curated Benchmark   │
│  - UMAP / PCA 2D Manifolds    │             │  - Concept Centroid Progression      │
│  - Medoid Exemplar Extraction │             │  - Empirical Solve Statistics Filter │
└───────────────┬───────────────┘             └──────────────────┬───────────────────┘
                │                                                │
                └───────────────────────┬────────────────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │     Streamlit 4-Tab Web UI    │
                        │  - Search & Discovery         │
                        │  - Side-by-Side Mode Contrast │
                        │  - 2D Interactive Plotly Map  │
                        │  - Difficulty Step & Code Peek│
                        └───────────────────────────────┘
```

---

## Empirical Benchmark Evaluation

Retrieval performance was evaluated using two complementary frameworks:
1. **Curated Hand-Crafted Benchmark (`data/eval_set.csv`):** 45 gold-standard competitive programming queries representing distinct algorithmic archetypes (Flow Networks, Knapsack DP, Fenwick Trees, Convex Hull, Palindrome Automata).
2. **Global Tag-Overlap Benchmark:** 200 randomly sampled test problems evaluated across topic tag co-occurrence.

### Retrieval Performance Matrix

| Model Architecture | Retrieval Mode | P@5 | MRR (Mean Reciprocal Rank) | nDCG@10 | Latency (CPU) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BM25 / TF-IDF Baseline** | Sparse Keyword | 0.062 | 0.186 | 0.133 | **1.2 ms** |
| **BAAI/bge-small-en-v1.5** | Dense Semantic | 0.044 | 0.166 | 0.105 | 24.1 ms |
| **all-MiniLM-L6-v2** | Dense Semantic | 0.062 | **0.217** *(+16.7%)* | 0.141 | 5.6 ms |
| **Hybrid Blend (α = 0.65)** | Hybrid (Dense + Sparse) | **0.067** | 0.200 | **0.147** *(+10.5%)* | 6.8 ms |

### Key Theoretical Insights
- **Dense vs. Sparse Lift:** Dense semantic representations delivered a **+16.7% improvement in MRR** over sparse keyword baselines by successfully matching underlying algorithmic mechanics across disparate narrative settings.
- **Model Architecture Selection:** `all-MiniLM-L6-v2` outperformed `bge-small-en-v1.5` by **+30.7% MRR** on competitive programming tasks while executing **4.3x faster** on CPU. Dense text models pre-trained heavily on short conversational pairs underperform sentence-transformers tuned for long-document paragraph similarity on structured STEM text.
- **Hybrid Calibration:** Tuning $\alpha$ over a fine grid revealed that an $\alpha = 0.65$ blend maximizes $nDCG@10$ ($0.147$), combining semantic understanding with exact constraint matching (e.g., $O(N \log N)$ keyword limits).

---

## Algorithmic Discoveries: "Hidden Matches"

A core capability of Déjà Code is detecting **Hidden Matches**—problem pairs with extremely high dense semantic affinity ($\cos \theta \ge 0.62$) despite near-zero lexical overlap ($\text{TF-IDF} \le 0.12$).

| Query Problem | Hidden Match | Semantic Sim | Keyword Sim | Algorithmic Rationale |
| :--- | :--- | :---: | :---: | :--- |
| **`LEBOMBS`** (Little Elephant and Bombs) | **`CHEFARRB`** (Chef and Array) | **0.785** | 0.047 | 1D sequential state traversal with contiguous neighborhood updates. |
| **`BUYING2`** (Banknotes and Coins) | **`ALEXNUMB`** (Magic Pairs) | **0.778** | 0.065 | Modulo-based pigeonhole principle and discrete combinatorics. |
| **`DEVJERRY`** (Devu and Jerry Mouse) | **`GRIDGAME`** (Game on a Grid) | **0.769** | 0.038 | Dynamic programming on a 2D coordinate grid with obstacle avoidance. |
| **`ANUSPM`** (Sereja and Permutations) | **`RRSTONE`** (Stone Pile Game) | **0.761** | 0.039 | Array parity invariance and periodic transformation cycles. |

---

## Error Analysis & Failure Taxonomy

Evaluating Top-1 semantic retrieval against human-annotated topic tags identified four distinct failure mechanisms across `out/failures.csv`:

```
                             [ Failure Modes Distribution ]
      ┌─────────────────────────────────────────────────────────────┐
      │  Shared Story Words (Entity Overlap)          48.0% (84)    │
      │  Noisy / Incomplete Ground Truth Tags         30.8% (54)    │
      │  Shared Input / Constraint Boilerplate        20.6% (36)    │
      │  Short Ambiguous Statements (< 180 chars)      0.6% (1)     │
      └─────────────────────────────────────────────────────────────┘
```

1. **Shared Story Words (48.0%):** Character names (*Chef, Sereja, Little Elephant*) dominate vector projections. Mitigated via domain-specific stopword filtering and higher hybrid $\alpha$.
2. **Noisy / Incomplete Ground Truth Tags (30.8%):** Retrieval was algorithmically correct (e.g. `Sereja and GCD` matching `Sereja and LCM`), but CodeChef metadata was missing the ground-truth topic tag.
3. **Shared Input / Constraint Boilerplate (20.6%):** Structural framing (*"The first line contains T test cases followed by N integers"*) skewed representations before regex stripping.
4. **Short Ambiguous Statements (0.6%):** Highly terse statements (< 180 characters) lacked sufficient contextual tokens for discriminative embedding.

---

## Adaptive Recommendation Engine

Déjà Code implements a concept-aligned difficulty progression engine:
1. **User Skill Profiling:** Solved problems $S = \{s_1, \dots, s_m\}$ are mapped to difficulty integers:
   $$\text{Level}_{\text{user}} = \left\lfloor \frac{1}{|S|} \sum_{s \in S} \text{Diff}(s) \right\rceil \in \{0: \text{Beginner}, 1: \text{Easy}, 2: \text{Medium}, 3: \text{Hard}, 4: \text{Challenge}\}$$
2. **Concept Centroid Extraction:** The user's demonstrated domain concept is captured by the normalized centroid:
   $$\bar{\mathbf{c}} = \frac{\sum_{s \in S} \mathbf{e}_s}{\left\|\sum_{s \in S} \mathbf{e}_s\right\|_2}$$
3. **Difficulty Stepping:** Targets tier $\text{Level}_{\text{target}} = \text{clip}(\text{Level}_{\text{user}} + \Delta, 0, 4)$ where $\Delta \in \{+1, 0, -1\}$.
4. **Ranking & Solve Stats Integration:** Unsolved candidate problems matching the target difficulty are ranked by cosine similarity to $\bar{\mathbf{c}}$ and enriched with empirical acceptance rates derived from 637k submissions:
   $$\text{Rank}(p) = \cos(\mathbf{e}_p, \bar{\mathbf{c}}) \quad \text{tie-broken by} \quad \frac{\text{Accepted}(p)}{\text{Submissions}(p)}$$

---

## Interactive Streamlit Application

The accompanying web interface provides an analytical workbench across 4 specialized modules:

- **Tab 1: Search & Lookup:** Natural language problem search with real-time hybrid alpha weighting, difficulty filters, and statement previews.
- **Tab 2: Mode Comparison:** Side-by-side contrast of Sparse Keyword vs Dense Semantic search on identical queries, computing real-time Jaccard overlap and novel semantic discovery counts.
- **Tab 3: Interactive Clusters:** Nonlinear 2D UMAP scatter projection built with Plotly Graph Objects. Includes cluster profile tables and medoid exemplar inspection.
- **Tab 4: Adaptive Recommendations:** Solved problem selection with progression ladder controls (`+1 Level Up`, `0 Reinforce`, `-1 Review`), empirical solve statistics, and an accepted code peek drawer.

```bash
streamlit run app.py
```

---

## Quickstart & CLI Reference

### 1. Environment Setup
```bash
git clone https://github.com/visshalb2007/dejacode.git
cd dejacode
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Pipeline Execution
```bash
# Execute full pipeline: clustering, evaluation, hidden matches, recommendations
python dejacode.py --data data/questions.csv --k 12 --cluster-method kmeans --dim-red umap

# Generate recommendations for specific solved problem codes
python dejacode.py --data data/questions.csv --recommend-solved FLOW017,CFRTEST --recommend-step 1

# Execute hyperparameter tuning and model comparisons
python dejacode.py --data data/questions.csv --tune
```

### CLI Arguments

| Flag | Default | Description |
| :--- | :---: | :--- |
| `--data` | *Required* | Path to input problems CSV (`data/questions.csv`) |
| `--k` | `12` | Number of clusters for K-Means |
| `--cluster-method` | `kmeans` | Clustering algorithm (`kmeans`, `hdbscan`) |
| `--dim-red` | `umap` | 2D dimensionality reduction manifold (`umap`, `pca`) |
| `--alpha` | `0.65` | Hybrid search dense score weighting coefficient |
| `--recommend-solved` | `FLOW017,CFRTEST` | Comma-separated QCodes for next problem recommendations |
| `--recommend-step` | `1` | Difficulty escalation step (`+1` next tier, `0` same tier) |
| `--tune` | `False` | Run alpha parameter sweep and dense model comparison |
| `--force-embed` | `False` | Force re-computation of embeddings ignoring cache |

---

## Repository Structure

```
dejacode/
├── dejacode.py              # End-to-end pipeline: ETL, embeddings, clustering, evaluation, recs
├── app.py                   # Multi-tab Streamlit analytical web application
├── requirements.txt         # Pinned production dependencies
├── data/
│   ├── questions.csv        # 1,429 CodeChef problem statements with difficulty & tags
│   ├── solutions.csv        # 637,763 code submissions metadata (indexed via chunking)
│   ├── eval_set.csv         # 45-query gold-standard algorithmic evaluation benchmark
│   └── program_codes/       # Accepted C++, Python, Java raw code submissions
├── out/                     # Generated artifacts & analytical summaries
│   ├── emb.npy              # 384-dimensional dense embeddings matrix
│   ├── coords_2d.npy        # 2D UMAP projection coordinates
│   ├── problems_clustered.csv# Clustered questions dataset with topic labels
│   ├── cluster_summary.csv  # Cluster profiles, dominant tags, and medoid exemplars
│   ├── hidden_matches.csv   # Top 30 algorithmically isomorphic problem pairs
│   ├── failures.csv         # Categorized error analysis and failure instances
│   ├── failure_summary.json # Aggregated taxonomy breakdown
│   ├── alpha_tuning.csv     # Grid search sweep of dense/sparse alpha weights
│   ├── model_tuning_results.csv # Empirical comparison of embedding models
│   └── recommendations.csv  # Recommended practice problems
└── tests/
    └── test_baseline.py     # Unit test suite verifying preprocessing, search, and clusters
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
