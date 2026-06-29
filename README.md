# Embedding Layers: From Lexical Matching to Semantic Meaning

## Table of Contents

1. [Purpose](#purpose)
2. [Conceptual Overview](#conceptual-overview)
3. [Lexical Layer](#lexical-layer)
4. [Semantic Layer](#semantic-layer)
5. [Distance Functions](#distance-functions)
6. [Hybrid Retrieval](#hybrid-retrieval)
7. [Query Preprocessing](#query-preprocessing)
8. [Vector Projection](#vector-projection)
9. [Dataset Strategy](#dataset-strategy)
10. [Evaluation and Assertions](#evaluation-and-assertions)
11. [Project Structure](#project-structure)
12. [Installation](#installation)
13. [Running the Streamlit PoC](#running-the-streamlit-poc)
14. [Running Tests](#running-tests)
15. [Suggested Experiments](#suggested-experiments)
16. [Limitations](#limitations)

## Purpose

This project is a proof of concept about embedding layers. Its central objective is to make the transition from lexical representations to semantic representations observable, reproducible, and pedagogically clear.

The project starts with lexical vectorization strategies, where documents are represented by the words or character patterns that literally occur in them. It then introduces semantic embeddings, where documents are represented by dense numerical vectors learned from language models. Finally, it combines both views through a hybrid retrieval layer.

The intended demonstration is simple: two documents may be lexically different and semantically similar at the same time. For instance, "heart attack" and "myocardial infarction" may share few surface tokens, but they refer to a closely related clinical concept. A purely lexical system may fail to retrieve this relation, while a semantic system may capture it.

## Conceptual Overview

An embedding is a function that maps an object, such as a text, into a vector space:

```text
f: text -> R^d
```

In retrieval systems, this vector space allows texts to be compared numerically. A query and a document can be embedded into the same space, and a distance or similarity function can estimate their relatedness.

This project separates the retrieval problem into three layers:

1. The lexical layer, which models surface-level token overlap.
2. The semantic layer, which models contextual meaning.
3. The hybrid layer, which combines lexical precision with semantic recall.

The Streamlit application lets the user explore each corpus, inspect labeled validation queries, choose strategies in each layer, and observe how rankings, scores, validation metrics, backend execution status, and two-dimensional vector projections change.

## Lexical Layer

The lexical layer represents documents using explicit textual evidence. It is interpretable because each dimension is derived from observable terms, n-grams, or character patterns. In this PoC, the searchable representation of each document is `title + text`, because titles often contain compact lexical signals that users reasonably expect to influence retrieval.

### Count Vectorization

Count vectorization represents each document by the frequency of each token:

```text
x_{d,t} = count(t, d)
```

where `x_{d,t}` is the value of term `t` in document `d`.

This strategy is simple and transparent. However, frequent but uninformative words may dominate the representation unless stopword filtering or weighting is applied.

### TF-IDF

Term Frequency-Inverse Document Frequency adjusts raw token counts by how rare each term is across the corpus:

```text
tfidf(t, d, D) = tf(t, d) * log(N / df(t))
```

where `tf(t, d)` is the frequency of term `t` in document `d`, `N` is the number of documents, and `df(t)` is the number of documents containing `t`.

TF-IDF gives more importance to terms that are frequent in a document but uncommon in the full corpus. It is a strong baseline for lexical retrieval.

### Word N-Grams

Word n-grams represent contiguous sequences of words. A unigram model uses single tokens, while a bigram model captures two-token phrases such as "social media" or "machine learning".

This project exposes configurable n-gram ranges so the user can compare unigram, bigram, and mixed representations.

### Character N-Grams

Character n-grams represent subword fragments. They are useful when texts contain misspellings, abbreviations, morphology, or noisy social media language.

For example, "diagnosis", "diagnostic", and "diagnosed" share character-level fragments even when their full tokens differ. The same principle explains why a typo such as "coffe" can still retrieve a document titled "Coffee enthusiasm": the query and the document share character fragments even though they do not share the exact word token "coffee".

### BM25

BM25 is a probabilistic lexical ranking function. It improves on raw frequency by introducing term-frequency saturation and document-length normalization:

```text
BM25(q, d) = sum_{t in q} IDF(t) * ((tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl)))
```

where `k1` controls term-frequency saturation, `b` controls length normalization, `|d|` is the document length, and `avgdl` is the average document length in the corpus.

BM25 is still lexical: it does not understand synonymy by itself. However, it is often a stronger sparse retrieval baseline than plain TF-IDF.

## Semantic Layer

The semantic layer uses dense embeddings produced by a sentence-transformer model. Instead of representing literal token overlap, it places texts with related meaning near each other in a vector space.

If a model maps two texts into vectors `u` and `v`, their semantic similarity can be estimated even when the texts use different vocabulary.

The default semantic model configured by the project is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model is compact enough for a PoC and widely used for sentence-level semantic similarity. The application also includes a deterministic hashing fallback, which keeps the PoC runnable when the transformer model is not available locally. The interface reports the requested backend and the backend actually used, so fallback behavior remains explicit rather than hidden.

## Distance Functions

### Cosine Similarity

Cosine similarity measures the angle between two vectors:

```text
cosine(u, v) = (u . v) / (||u|| ||v||)
```

It is commonly used in text retrieval because vector direction is often more important than vector magnitude. Two documents can have different lengths but still point toward a similar semantic or lexical direction.

Cosine distance is defined as:

```text
cosine_distance(u, v) = 1 - cosine(u, v)
```

### Euclidean Distance

Euclidean distance measures the straight-line distance between two vectors:

```text
euclidean(u, v) = sqrt(sum_i (u_i - v_i)^2)
```

In this project, Euclidean distance is converted into a similarity score:

```text
similarity(u, v) = 1 / (1 + euclidean(u, v))
```

This keeps scores in a ranking-friendly direction: larger values mean stronger similarity. The PoC includes two dedicated Euclidean corpora. `Distance metric laboratory` contains documents such as `coffee` and `coffee coffee coffee`, so users can observe that cosine similarity treats them as the same direction while Euclidean similarity penalizes the repeated document because its count vector has a different magnitude. `Euclidean noise laboratory` contains compact matches such as `coffee remote` and noisy variants that add unrelated terms, showing that Euclidean distance also penalizes extra vector dimensions.

## Vector Projection

The interface includes a two-dimensional semantic vector map. It uses Principal Component Analysis (PCA) to project the current query vector and the top semantic document vectors into a plane:

```text
z = PCA_2(x)
```

This projection is not a ranking algorithm. It is an explanatory visualization that helps users see whether the query and retrieved documents are near each other in the selected semantic vector space. Because PCA compresses a high-dimensional vector into two dimensions, it should be read as an intuition aid rather than as a complete measurement of similarity.

## Hybrid Retrieval

Hybrid retrieval combines lexical and semantic scores:

```text
score(q, d) = alpha * lexical(q, d) + (1 - alpha) * semantic(q, d)
```

where `alpha` controls the importance of the lexical layer. A larger `alpha` prioritizes exact textual evidence. A smaller `alpha` prioritizes semantic meaning.

This is useful because lexical and semantic systems fail in different ways. Lexical retrieval is precise when terms match directly. Semantic retrieval is more robust when synonyms, paraphrases, or domain-specific concepts appear.

## Query Preprocessing

Query preprocessing is intentionally exposed as a separate option in the interface. This prevents the system from silently hiding the difference between a retrieval strategy and an input-normalization strategy.

The current preprocessing option is corpus-based spell correction. It builds a vocabulary from the selected corpus and maps unknown query tokens to close vocabulary matches. For example, in the social media corpus, the query:

```text
coffe
```

can be corrected to:

```text
coffee
```

This allows word-based lexical methods such as TF-IDF unigrams to recover a match that would otherwise be zero. However, this should be interpreted carefully: spell correction can improve retrieval for typographical errors, but it may also introduce incorrect substitutions in domain-specific text, abbreviations, names, slang, or clinical terminology.

Character n-grams solve a related problem differently. They do not correct the query; instead, they make the vectorization itself tolerant to partial character overlap. Therefore, spell correction and character n-grams are complementary preprocessing and representation strategies.

## Dataset Strategy

The project starts with controlled corpora because evaluation metrics require relevance judgments. Metrics such as P@N and MAP are only meaningful when the project knows which documents should be relevant for each query. Small hand-labeled corpora make the educational behavior explicit and reproducible.

For a larger version of the PoC, the controlled corpora can be complemented with external datasets. Good candidates include:

- `sklearn.datasets.fetch_20newsgroups` for general-topic documents.
- Hugging Face `datasets`, currently exposed through AG News examples for topical cosine/semantic search and Euclidean length-sensitivity experiments.
- BEIR-style information retrieval datasets for benchmark-oriented search evaluation.
- PubMed or biomedical abstract datasets for clinical and scientific terminology, subject to dataset licensing and access constraints.
- MIMIC-style clinical notes only when credentialing, privacy, and data-use requirements are satisfied.

The recommended architecture is to keep the current controlled corpora as deterministic teaching examples and add dataset loaders as a separate module. This keeps the PoC explainable while still allowing larger, realistic experiments.

## Evaluation and Assertions

The project includes validation metrics for each strategy. A small labeled dataset defines which documents are relevant for each query. The application computes metrics for lexical, semantic, and hybrid rankings.

### Precision at N

Precision at N measures how many of the top `N` retrieved documents are relevant:

```text
P@N = relevant_documents_in_top_N / N
```

It answers the question: "Among the first results shown to the user, how many are actually useful?"

### Average Precision

Average Precision evaluates the full ranking by averaging precision values at the positions where relevant documents appear:

```text
AP = (1 / R) * sum_k P@k * rel(k)
```

where `R` is the total number of relevant documents, `P@k` is precision at rank `k`, and `rel(k)` is `1` when the document at rank `k` is relevant.

### Mean Average Precision

Mean Average Precision is the mean of Average Precision over multiple queries:

```text
MAP = (1 / Q) * sum_q AP(q)
```

It is useful when comparing retrieval strategies across a benchmark of several queries.

### Assertions

The test suite asserts basic retrieval properties, metric correctness, and score normalization behavior. These assertions ensure that the PoC remains educational and technically coherent as new strategies are added.

## Project Structure

```text
embedding-layers/
  app/
    streamlit_app.py
  .streamlit/
    config.toml
  src/
    embedding_layers/
      corpus.py
      dataset_sources.py
      distances.py
      evaluation.py
      hybrid.py
      lexical.py
      preprocessing.py
      projection.py
      ranking.py
      semantic.py
  tests/
    test_dataset_sources.py
    test_evaluation.py
    test_preprocessing.py
    test_retrieval.py
  pyproject.toml
  README.md
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project:

```bash
pip install -e ".[dev]"
```

Install optional downloadable dataset support:

```bash
pip install -e ".[dev,datasets]"
```

The first use of the sentence-transformer model may require downloading model weights. If the model is unavailable, the Streamlit app can use the deterministic hashing fallback for demonstration purposes.

The project includes `.streamlit/config.toml` with Streamlit file watching disabled. This avoids noisy optional imports from the `transformers` package, such as image-processing modules that require `torchvision`, even though this project only uses text embeddings.

## Running the Streamlit PoC

Start the application:

```bash
streamlit run app/streamlit_app.py
```

The interface allows the user to select a corpus, choose a labeled query preset, enter a custom query, choose query preprocessing, choose lexical and semantic strategies, choose cosine or Euclidean similarity, adjust the hybrid weight, and inspect rankings and validation metrics. It can also download compact external samples from the same `Dataset` selector: `20 Newsgroups` through scikit-learn, `AG News topical sample` through Hugging Face datasets (`fancyzhx/ag_news`) for cosine/semantic experiments, and `AG News Euclidean length sample` through Hugging Face datasets (`fancyzhx/ag_news`) for Euclidean length and noise experiments. A visual retrieval path is rendered before the result tabs, showing the original query, the effective query after preprocessing, the lexical layer, the semantic layer, the actual semantic backend used, the similarity function, observed scores for the current query, the filled hybrid scoring formula, and a PCA-based semantic vector map. The corpus explorer is available as the final tab so the main retrieval flow remains the first visible experience. For the social media corpus, the default lexical strategy is character n-grams because informal text often contains abbreviations and spelling noise.

## Running Tests

Run the test suite:

```bash
pytest
```

Run linting:

```bash
ruff check .
```

## Suggested Experiments

1. Compare "heart attack" with "myocardial infarction" in the clinical corpus.
2. Compare misspelled or abbreviated social media expressions with character n-grams.
3. Use the `Distance metric laboratory` corpus with query `coffee`, Count vectorizer, and Euclidean similarity to observe magnitude sensitivity.
4. Use the `Euclidean noise laboratory` corpus with query `coffee remote`, Count vectorizer, and Euclidean similarity to observe the effect of extra unrelated dimensions.
5. Switch both Euclidean laboratory queries from Euclidean similarity to cosine similarity and compare how direction differs from straight-line distance.
6. Increase the hybrid lexical weight and observe when exact token matching becomes dominant.
7. Add domain-specific corpora and measure whether MAP improves or degrades.

## Limitations

This project is a PoC, not a production retrieval engine. Its corpora are small, its evaluation labels are illustrative, and its semantic model is selected for accessibility rather than domain specialization.

A production system would require larger corpora, stronger evaluation protocols, model monitoring, latency measurements, versioned embeddings, and domain-specific relevance judgments.
