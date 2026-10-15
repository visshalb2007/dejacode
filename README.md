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
7. **Error analysis** - semantic top-1 results sharing no tag with the query go to `out/failures.csv` with categorized failure causes.

## Error Analysis & Failure Taxonomy

When evaluating Top-1 semantic retrieval against tag-overlap ground truth, false matches in `out/failures.csv` group into four distinct failure modes:

| Failure Cause | Frequency | Share | Primary Mechanism |
| :--- | :---: | :---: | :--- |
| **Shared Story Words** | 84 | 48.0% | Strong narrative vocabulary overlap (character names, recipes, shop/coins) dominates embedding distance over algorithmic cues. |
| **Noisy / Missing Tags** | 54 | 30.8% | Retrieval was conceptually sound, but the candidate problem has incomplete tags in CodeChef metadata (false negative in benchmark). |
| **Shared Input Format** | 36 | 20.6% | Identical constraint and I/O boilerplate ("first line contains T...", "consisting of N integers") skews representations. |
| **Short Ambiguous Statements** | 1 | 0.6% | Minimal text length (< 180 chars) lacks sufficient contextual tokens for dense discrimination. |

### 5 Concrete Case Studies

1. **Shared Story Words (Entity Overfitting):**
   - **Query:** `Little Elephant and Balloons` (Tags: `['dp']`)
   - **Retrieved:** `Little Elephant and Boxes` (Score: 0.736, Tags: `[]`)
   - **Diagnosis:** Both problems belong to the "Little Elephant" story series. The encoder places heavy attention weights on the entity "Little Elephant", retrieving a different problem from the same storyline rather than a problem sharing dynamic programming mechanics.

2. **Shared Story Words (Domain Narrative Distraction):**
   - **Query:** `Optimal Shopping II` (Tags: `['travel-salesman']`)
   - **Retrieved:** `Suraj goes shopping` (Score: 0.605, Tags: `['greedy']`)
   - **Diagnosis:** Both problem descriptions revolve around shopping in retail stores, budget limits, and selecting items. The model was distracted by the shopping narrative instead of recognizing that Problem 1 is NP-hard Traveling Salesperson while Problem 2 is a simple greedy sorting problem.

3. **Noisy Tags (Missing Ground-Truth in Candidate):**
   - **Query:** `Sereja and GCD` (Tags: `['number-theory']`)
   - **Retrieved:** `Sereja and LCM` (Score: 0.876, Tags: `['dp', 'matrix-expo']`)
   - **Diagnosis:** Both problems compute gcd/lcm properties of sequences under modular arithmetic. The semantic match is algorithmically legitimate, but `Sereja and LCM` was never tagged with `number-theory` in CodeChef metadata, incorrectly registering as an evaluation failure.

4. **Shared Input / Constraint Formatting:**
   - **Query:** `Strange operations` (Tags: `['basic-prog', 'simulation']`)
   - **Retrieved:** `Code` (Score: 0.687, Tags: `[]`)
   - **Diagnosis:** Both problem statements are short and dominated by standard CodeChef test case templates ("The first line contains T test cases... followed by N integers"). The structural template similarity overshadowed the actual problem logic.

5. **Short / Ambiguous Statements:**
   - **Query:** `Lyra and the Alethiometer` (Tags: `['adhoc']`, Statement length: 142 chars)
   - **Retrieved:** `Superpowers of 2` (Score: 0.353, Tags: `[]`)
   - **Diagnosis:** The prompt contained barely three sentences of text. Without sufficient descriptive tokens, the resulting embedding lacked discriminative power, producing low cosine scores and near-arbitrary retrieval.

## Limitations
- Tag overlap is a noisy relevance proxy; many problems have few or no topic tags.
- KMeans needs k fixed in advance and assumes round clusters.
- MiniLM may over-weight story words; a code-aware model may do better.
