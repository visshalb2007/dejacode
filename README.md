# Déjà Code - Semantic Search & Clustering for Competitive Programming Problems

Finds programming problems that share the same underlying idea, even when the story is different.
Dataset: CodeChef Competitive Programming (`data/questions.csv`, ~1,470 problems).

## Project layout
    dejacode/
      dejacode.py        # pipeline: preprocess, embed, search, cluster, evaluate
      app.py             # Streamlit demo
      requirements.txt
      data/questions.csv # dataset (included)
      out/               # generated: embeddings, clusters, plot, metrics, failures

## Run
    pip install -r requirements.txt
    python dejacode.py --data data/questions.csv --k 10
    streamlit run app.py

## Pipeline
1. **Preprocess** - strips HTML, LaTeX, URLs, "Input/Output" labels and CodeChef boilerplate ("All submissions...", "Read problems statements in..."); drops scraped website-junk rows, very short and duplicate statements.
2. **Tag cleaning** - keeps topic tags only; removes author names, contest codes (cook64, jan15) and difficulty words; merges synonyms (dp / dynamic-prog, math / maths).
3. **Embeddings** - Sentence-BERT `all-MiniLM-L6-v2` on "title + statement" (normalized, dot product = cosine). Swap with `--model`.
4. **Search** - semantic (cosine), keyword (TF-IDF baseline), hybrid (z-scored blend, alpha=0.7).
5. **Clustering** - KMeans on embeddings; clusters labelled by top TF-IDF terms; PCA plot in `out/clusters.png`.
6. **Evaluation** - 200 sampled problems as queries; a result is relevant if it shares >=1 topic tag. P@10, MRR, nDCG@10 for keyword vs semantic vs hybrid (`out/metrics.csv`).
7. **Error analysis** - semantic top-1 results sharing no tag with the query go to `out/failures.csv`.

## Limitations
- Tag overlap is a noisy relevance proxy; many problems have few or no topic tags.
- KMeans needs k fixed in advance and assumes round clusters.
- MiniLM may over-weight story words; a code-aware model may do better.

## Not built yet
HDBSCAN, UMAP, FAISS, LLM-generated cluster labels, difficulty-based recommendations, interactive explorer.
