"""Streamlit interface for the embedding layers PoC."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from embedding_layers.corpus import CORPORA, Corpus, get_corpus_by_name
from embedding_layers.dataset_sources import (
    load_20newsgroups_sample,
    load_ag_news_cosine_sample,
    load_ag_news_euclidean_length_sample,
)
from embedding_layers.evaluation import EvaluationSummary, evaluate_strategy
from embedding_layers.hybrid import retrieve_hybrid
from embedding_layers.lexical import LEXICAL_STRATEGIES, retrieve_lexical
from embedding_layers.preprocessing import (
    correct_query_with_corpus_vocabulary,
    tokenize_text,
)
from embedding_layers.projection import build_projection_frame
from embedding_layers.ranking import RankedResult
from embedding_layers.semantic import (
    SEMANTIC_STRATEGIES,
    SemanticBackendStatus,
    retrieve_semantic,
    retrieve_semantic_with_status,
)

DISTANCE_OPTIONS = ("Cosine similarity", "Euclidean similarity")
PREPROCESSING_OPTIONS = ("No preprocessing", "Corpus spell correction")
DOWNLOAD_20_NEWSGROUPS_LABEL = "Download 20 Newsgroups sample"
DOWNLOAD_AG_NEWS_COSINE_LABEL = "Download AG News topical sample"
DOWNLOAD_AG_NEWS_EUCLIDEAN_LABEL = "Download AG News Euclidean length sample"
EUCLIDEAN_LAB_CORPORA = (
    "Distance metric laboratory",
    "Euclidean noise laboratory",
    "AG News Euclidean length sample",
)


@st.cache_data(show_spinner=False)
def _load_20newsgroups_sample() -> Corpus:
    """Load and cache the optional 20 Newsgroups sample corpus."""

    return load_20newsgroups_sample()


@st.cache_data(show_spinner=False)
def _load_ag_news_cosine_sample() -> Corpus:
    """Load and cache the optional AG News topical corpus."""

    return load_ag_news_cosine_sample()


@st.cache_data(show_spinner=False)
def _load_ag_news_euclidean_length_sample() -> Corpus:
    """Load and cache the optional AG News Euclidean corpus."""

    return load_ag_news_euclidean_length_sample()


def _results_to_frame(results: list[RankedResult]) -> pd.DataFrame:
    """Convert ranked results into a display dataframe."""

    rows = []
    for result in results:
        row = {
            "rank": result.rank,
            "id": result.document.identifier,
            "title": result.document.title,
            "score": round(result.score, 4),
            "text": result.document.text,
        }
        if result.lexical_score is not None:
            row["lexical_score"] = round(result.lexical_score, 4)
        if result.semantic_score is not None:
            row["semantic_score"] = round(result.semantic_score, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def _documents_to_frame(corpus: Corpus, search_text: str) -> pd.DataFrame:
    """Convert corpus documents into a filterable display dataframe."""

    normalized_search = search_text.strip().lower()
    rows = []
    for document in corpus.documents:
        searchable_text = document.searchable_text
        if normalized_search and normalized_search not in searchable_text.lower():
            continue
        rows.append(
            {
                "id": document.identifier,
                "title": document.title,
                "domain": document.domain,
                "searchable_text": searchable_text,
            }
        )
    return pd.DataFrame(rows)


def _query_cases_to_frame(corpus: Corpus) -> pd.DataFrame:
    """Convert labeled query cases into a display dataframe."""

    return pd.DataFrame(
        {
            "query": query_case.query,
            "relevant_document_ids": ", ".join(
                sorted(query_case.relevant_document_ids)
            ),
        }
        for query_case in corpus.query_cases
    )


def _evaluation_to_frame(summary_name: str, summary: EvaluationSummary) -> pd.DataFrame:
    """Convert evaluation metrics into a display dataframe."""

    return pd.DataFrame(
        {
            "strategy": summary_name,
            "query": item.query,
            "P@N": round(item.precision_at_n, 4),
            "AP": round(item.average_precision, 4),
        }
        for item in summary.query_evaluations
    )


def _strategy_description(strategy_name: str) -> str:
    """Return a short strategy description for the selected option."""

    for strategy in (*LEXICAL_STRATEGIES, *SEMANTIC_STRATEGIES):
        if strategy.name == strategy_name:
            return strategy.description
    return "Selected strategy."


def _semantic_backend_note(status: SemanticBackendStatus) -> str:
    """Return a transparency note for the semantic backend actually used."""

    if status.actual_backend == "Deterministic hashing fallback":
        base_note = (
            "This execution used deterministic dense vectors from token hashes. "
            "It is useful for offline mechanics, but it is not a learned "
            "semantic language model."
        )
    else:
        base_note = "This execution used learned SentenceTransformer embeddings."

    if status.used_fallback:
        return (
            f"Requested `{status.requested_strategy}`, but used "
            f"`{status.actual_backend}`. {base_note}"
        )
    return f"Requested and used `{status.actual_backend}`. {base_note}"


def _lexical_formula(strategy_name: str) -> str:
    """Return the academic formula for a lexical strategy."""

    if strategy_name == "Count vectorizer":
        return r"x_{d,t} = count(t, d)"
    if strategy_name.startswith("TF-IDF"):
        return r"tfidf(t,d,D)=tf(t,d)\times log\left(\frac{N}{df(t)}\right)"
    if strategy_name == "BM25":
        return (
            r"BM25(q,d)=\sum_{t\in q} IDF(t)"
            r"\frac{tf(t,d)(k_1+1)}{tf(t,d)+k_1(1-b+b|d|/avgdl)}"
        )
    return r"x = vectorize(text)"


def _semantic_formula(strategy_name: str) -> str:
    """Return the academic formula for a semantic strategy."""

    if strategy_name == "Deterministic hashing fallback":
        return r"f(text) = normalize(hash\_features(text))"
    return r"f(text) = SentenceTransformer(text) \in \mathbb{R}^{384}"


def _distance_formula(distance_name: str) -> str:
    """Return the academic formula for a distance strategy."""

    if distance_name == "Cosine similarity":
        return r"cosine(u,v)=\frac{u\cdot v}{||u||\,||v||}"
    return r"similarity(u,v)=\frac{1}{1+\sqrt{\sum_i(u_i-v_i)^2}}"


def _word_overlap(query: str, document_text: str) -> list[str]:
    """Return shared word tokens between a query and a document."""

    query_tokens = set(tokenize_text(query))
    document_tokens = set(tokenize_text(document_text))
    return sorted(query_tokens & document_tokens)


def _character_ngrams(text: str, min_n: int = 3, max_n: int = 5) -> set[str]:
    """Return character n-grams used for transparent typo explanations."""

    normalized = f" {text.lower()} "
    ngrams: set[str] = set()
    for n_value in range(min_n, max_n + 1):
        for index in range(max(len(normalized) - n_value + 1, 0)):
            ngrams.add(normalized[index : index + n_value])
    return ngrams


def _lexical_evidence(query: str, result: RankedResult, strategy_name: str) -> str:
    """Describe why the top lexical document received its score."""

    if strategy_name == "TF-IDF character n-grams":
        overlap = sorted(
            _character_ngrams(query)
            & _character_ngrams(result.document.searchable_text)
        )
        sample = ", ".join(repr(item) for item in overlap[:8])
        return (
            f"{len(overlap)} shared character n-grams. "
            f"Sample: {sample or 'none'}."
        )

    overlap = _word_overlap(query, result.document.searchable_text)
    return (
        "Shared word tokens with top document: "
        f"{', '.join(overlap) or 'none'}."
    )


def _has_non_zero_score(results: list[RankedResult]) -> bool:
    """Return whether at least one ranked result has a positive score."""

    return any(result.score > 0.0 for result in results)


def _default_lexical_strategy_index(corpus: Corpus) -> int:
    """Return the default lexical strategy index for a corpus."""

    strategy_names = [strategy.name for strategy in LEXICAL_STRATEGIES]
    if corpus.name == "Social media language":
        return strategy_names.index("TF-IDF character n-grams")
    if corpus.name in EUCLIDEAN_LAB_CORPORA:
        return strategy_names.index("Count vectorizer")
    return strategy_names.index("TF-IDF word unigrams")


def _default_distance_strategy_index(corpus: Corpus) -> int:
    """Return the default distance strategy index for a corpus."""

    if corpus.name in EUCLIDEAN_LAB_CORPORA:
        return DISTANCE_OPTIONS.index("Euclidean similarity")
    return DISTANCE_OPTIONS.index("Cosine similarity")


def _preprocess_query(query: str, corpus: Corpus, preprocessing_name: str) -> str:
    """Apply the selected query preprocessing strategy."""

    if preprocessing_name == "Corpus spell correction":
        return correct_query_with_corpus_vocabulary(query, corpus.documents)
    return query


def _render_retrieval_path(
    query: str,
    effective_query: str,
    preprocessing_name: str,
    lexical_strategy: str,
    semantic_strategy: str,
    semantic_backend_status: SemanticBackendStatus,
    distance_name: str,
    lexical_weight: float,
    lexical_results: list[RankedResult],
    semantic_results: list[RankedResult],
    hybrid_results: list[RankedResult],
) -> None:
    """Render a transparent explanation of the current retrieval execution."""

    top_lexical = lexical_results[0]
    top_semantic = semantic_results[0]
    top_hybrid = hybrid_results[0]
    query_tokens = tokenize_text(effective_query)
    preprocessing_summary = (
        "No token changes were applied."
        if query == effective_query
        else f"The query was normalized from `{query}` to `{effective_query}`."
    )

    st.subheader("Current query execution")
    flow_frame = pd.DataFrame(
        [
            {
                "step": "1. Input query",
                "value": query,
                "what happened": "The text entered by the user.",
            },
            {
                "step": "2. Query preprocessing",
                "value": effective_query,
                "what happened": f"{preprocessing_name}. {preprocessing_summary}",
            },
            {
                "step": "3. Searchable document text",
                "value": "title + text",
                "what happened": (
                    "Each document is indexed using its title and body text."
                ),
            },
            {
                "step": "4. Lexical layer",
                "value": lexical_strategy,
                "what happened": _strategy_description(lexical_strategy),
            },
            {
                "step": "5. Semantic layer",
                "value": semantic_strategy,
                "what happened": (
                    f"{_strategy_description(semantic_strategy)} "
                    f"{_semantic_backend_note(semantic_backend_status)}"
                ),
            },
            {
                "step": "6. Similarity and ranking",
                "value": distance_name,
                "what happened": "Higher scores are ranked first in the result tables.",
            },
        ]
    )
    st.dataframe(flow_frame, width="stretch", hide_index=True)

    metric_columns = st.columns(4)
    metric_columns[0].metric("Effective query tokens", len(query_tokens))
    metric_columns[1].metric("Top lexical score", f"{top_lexical.score:.4f}")
    metric_columns[2].metric("Top semantic score", f"{top_semantic.score:.4f}")
    metric_columns[3].metric("Top hybrid score", f"{top_hybrid.score:.4f}")

    if semantic_backend_status.used_fallback:
        st.warning(
            "Semantic backend fallback occurred. The ranking is still useful "
            "for demonstrating the mechanics, but it should not be interpreted "
            "as contextual language-model retrieval."
        )

    explanation_frame = pd.DataFrame(
        [
            {
                "layer": "Lexical",
                "top document": top_lexical.document.title,
                "score": round(top_lexical.score, 4),
                "evidence": _lexical_evidence(
                    effective_query,
                    top_lexical,
                    lexical_strategy,
                ),
            },
            {
                "layer": "Semantic",
                "top document": top_semantic.document.title,
                "score": round(top_semantic.score, 4),
                "evidence": _semantic_backend_note(semantic_backend_status),
            },
            {
                "layer": "Hybrid",
                "top document": top_hybrid.document.title,
                "score": round(top_hybrid.score, 4),
                "evidence": (
                    f"lexical={top_hybrid.lexical_score:.4f}, "
                    f"semantic={top_hybrid.semantic_score:.4f}, "
                    f"alpha={lexical_weight:.2f}"
                ),
            },
        ]
    )
    st.dataframe(explanation_frame, width="stretch", hide_index=True)

    st.subheader("Formulas applied to this query")
    formula_left, formula_right = st.columns(2)
    with formula_left:
        st.markdown("**Lexical representation**")
        st.latex(_lexical_formula(lexical_strategy))
        st.caption(
            f"For q = `{effective_query}`, top lexical document = "
            f"`{top_lexical.document.title}`, score = {top_lexical.score:.4f}."
        )
    with formula_right:
        st.markdown("**Semantic representation**")
        st.latex(_semantic_formula(semantic_strategy))
        st.caption(
            f"For q = `{effective_query}`, top semantic document = "
            f"`{top_semantic.document.title}`, score = {top_semantic.score:.4f}."
        )

    formula_left, formula_right = st.columns(2)
    with formula_left:
        st.markdown("**Similarity function**")
        st.latex(_distance_formula(distance_name))
        st.caption(f"Current distance strategy: `{distance_name}`.")
    with formula_right:
        st.markdown("**Hybrid score for the top hybrid result**")
        lexical_score = top_hybrid.lexical_score or 0.0
        semantic_score = top_hybrid.semantic_score or 0.0
        st.latex(
            rf"{top_hybrid.score:.4f}="
            rf"{lexical_weight:.2f}\cdot {lexical_score:.4f}+"
            rf"{1.0 - lexical_weight:.2f}\cdot {semantic_score:.4f}"
        )
        st.caption(f"Top hybrid document: `{top_hybrid.document.title}`.")


def _render_semantic_vector_map(projection_frame: pd.DataFrame) -> None:
    """Render a two-dimensional semantic vector projection."""

    st.subheader("Semantic vector map")
    st.caption(
        "PCA projects the current query vector and the top semantic documents "
        "into two dimensions. Points close together are near in the selected "
        "semantic vector space; this is an explanatory projection, not a new "
        "ranking algorithm."
    )
    st.scatter_chart(
        projection_frame,
        x="x",
        y="y",
        color="kind",
        size="size",
        width="stretch",
    )
    st.dataframe(
        projection_frame[["label", "kind", "score", "x", "y"]],
        width="stretch",
        hide_index=True,
    )


def _downloadable_dataset_loaders():
    """Return labels and loaders for optional downloadable datasets."""

    return {
        DOWNLOAD_20_NEWSGROUPS_LABEL: (
            _load_20newsgroups_sample,
            "Downloads via scikit-learn on first use, then caches locally.",
        ),
        DOWNLOAD_AG_NEWS_COSINE_LABEL: (
            _load_ag_news_cosine_sample,
            "Downloads AG News via Hugging Face datasets on first use.",
        ),
        DOWNLOAD_AG_NEWS_EUCLIDEAN_LABEL: (
            _load_ag_news_euclidean_length_sample,
            "Downloads AG News via Hugging Face datasets for Euclidean tests.",
        ),
    }


def _select_corpus() -> Corpus:
    """Select a corpus or an optional downloaded dataset directly."""

    downloadable_loaders = _downloadable_dataset_loaders()
    dataset_options = [corpus.name for corpus in CORPORA]
    dataset_options.extend(downloadable_loaders.keys())
    selected_dataset = st.sidebar.selectbox("Dataset", dataset_options)

    if selected_dataset in downloadable_loaders:
        loader, caption = downloadable_loaders[selected_dataset]
        st.sidebar.caption(caption)
        try:
            with st.spinner(f"Loading {selected_dataset}..."):
                return loader()
        except Exception as exc:
            st.sidebar.error(f"Dataset download failed: {exc}")
            st.sidebar.info("Falling back to the built-in academic corpus.")
            return get_corpus_by_name("Academic abstracts")

    return get_corpus_by_name(selected_dataset)


def main() -> None:
    """Run the Streamlit application."""

    st.set_page_config(
        page_title="Embedding Layers",
        page_icon="EL",
        layout="wide",
    )
    st.title("Embedding Layers: From Words to Meaning")

    corpus = _select_corpus()
    query_options = [query_case.query for query_case in corpus.query_cases]
    selected_query = st.sidebar.selectbox("Query preset", query_options)
    lexical_strategy = st.sidebar.selectbox(
        "Lexical strategy",
        [strategy.name for strategy in LEXICAL_STRATEGIES],
        index=_default_lexical_strategy_index(corpus),
    )
    semantic_strategy = st.sidebar.selectbox(
        "Semantic strategy",
        [strategy.name for strategy in SEMANTIC_STRATEGIES],
    )
    preprocessing_name = st.sidebar.selectbox(
        "Query preprocessing",
        PREPROCESSING_OPTIONS,
    )
    distance_name = st.sidebar.selectbox(
        "Distance function",
        DISTANCE_OPTIONS,
        index=_default_distance_strategy_index(corpus),
    )
    lexical_weight = st.sidebar.slider(
        "Hybrid lexical weight",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )
    n_value = st.sidebar.number_input(
        "Validation N",
        min_value=1,
        max_value=len(corpus.documents),
        value=min(3, len(corpus.documents)),
    )

    query = st.text_input("Query", value=selected_query)
    effective_query = _preprocess_query(query, corpus, preprocessing_name)

    lexical_results = retrieve_lexical(
        query=effective_query,
        documents=corpus.documents,
        strategy_name=lexical_strategy,
        distance_name=distance_name,
    )
    semantic_output = retrieve_semantic_with_status(
        query=effective_query,
        documents=corpus.documents,
        strategy_name=semantic_strategy,
        distance_name=distance_name,
    )
    semantic_results = semantic_output.results
    hybrid_results = retrieve_hybrid(
        query=effective_query,
        documents=corpus.documents,
        lexical_strategy_name=lexical_strategy,
        semantic_strategy_name=semantic_strategy,
        distance_name=distance_name,
        lexical_weight=lexical_weight,
    )

    st.caption(corpus.description)
    if corpus.name == "Distance metric laboratory":
        st.info(
            "This corpus is designed for Euclidean distance. With Count vectorizer, "
            "documents may point in the same lexical direction but have different "
            "magnitudes because terms are repeated different numbers of times. "
            "Try query `coffee`: cosine gives the exact and repeated coffee "
            "documents the same score, while Euclidean penalizes the repeated "
            "document because its count vector is farther from the query vector."
        )
    if corpus.name == "Euclidean noise laboratory":
        st.info(
            "This corpus is designed for Euclidean distance under lexical noise. "
            "Try query `coffee remote`: documents that include those terms plus "
            "many unrelated words gain extra vector dimensions, increasing their "
            "straight-line distance from the compact query vector."
        )
    if corpus.name == "AG News Euclidean length sample":
        st.info(
            "This downloaded corpus is useful for Euclidean distance because real "
            "news documents have different lengths and extra terms. With Count "
            "vectorizer, Euclidean similarity is sensitive to those magnitude and "
            "noise differences."
        )
    metric_left, metric_middle, metric_right = st.columns(3)
    metric_left.metric("Documents", len(corpus.documents))
    metric_middle.metric("Validation queries", len(corpus.query_cases))
    metric_right.metric("Hybrid lexical weight", f"{lexical_weight:.2f}")
    _render_retrieval_path(
        query=query,
        effective_query=effective_query,
        preprocessing_name=preprocessing_name,
        lexical_strategy=lexical_strategy,
        semantic_strategy=semantic_strategy,
        semantic_backend_status=semantic_output.backend_status,
        distance_name=distance_name,
        lexical_weight=lexical_weight,
        lexical_results=lexical_results,
        semantic_results=semantic_results,
        hybrid_results=hybrid_results,
    )

    projection_frame = build_projection_frame(
        query=effective_query,
        documents=corpus.documents,
        vectors=semantic_output.embeddings,
        ranked_results=semantic_results,
    )
    _render_semantic_vector_map(projection_frame)

    lexical_tab, semantic_tab, hybrid_tab, validation_tab, corpus_tab = st.tabs(
        ["Lexical", "Semantic", "Hybrid", "Validation", "Corpus"]
    )

    with lexical_tab:
        st.subheader("Lexical ranking")
        if not _has_non_zero_score(lexical_results):
            st.warning(
                "The selected lexical strategy produced only zero scores. "
                "For misspellings such as 'coffe' instead of 'coffee', use "
                "Corpus spell correction or switch to TF-IDF character n-grams. "
                "Word-based methods require exact token overlap."
            )
        st.dataframe(_results_to_frame(lexical_results), width="stretch")

    with semantic_tab:
        st.subheader("Semantic ranking")
        st.dataframe(_results_to_frame(semantic_results), width="stretch")

    with hybrid_tab:
        st.subheader("Hybrid ranking")
        st.dataframe(_results_to_frame(hybrid_results), width="stretch")

    with validation_tab:
        lexical_summary = evaluate_strategy(
            corpus,
            "Lexical",
            lambda validation_query: retrieve_lexical(
                _preprocess_query(validation_query, corpus, preprocessing_name),
                corpus.documents,
                lexical_strategy,
                distance_name,
            ),
            int(n_value),
        )
        semantic_summary = evaluate_strategy(
            corpus,
            "Semantic",
            lambda validation_query: retrieve_semantic(
                _preprocess_query(validation_query, corpus, preprocessing_name),
                corpus.documents,
                semantic_strategy,
                distance_name,
            ),
            int(n_value),
        )
        hybrid_summary = evaluate_strategy(
            corpus,
            "Hybrid",
            lambda validation_query: retrieve_hybrid(
                _preprocess_query(validation_query, corpus, preprocessing_name),
                corpus.documents,
                lexical_strategy,
                semantic_strategy,
                distance_name,
                lexical_weight,
            ),
            int(n_value),
        )
        overview = pd.DataFrame(
            [
                {
                    "strategy": summary.strategy_name,
                    "mean P@N": round(summary.precision_at_n, 4),
                    "MAP": round(summary.mean_average_precision, 4),
                }
                for summary in (lexical_summary, semantic_summary, hybrid_summary)
            ]
        )
        details = pd.concat(
            [
                _evaluation_to_frame("Lexical", lexical_summary),
                _evaluation_to_frame("Semantic", semantic_summary),
                _evaluation_to_frame("Hybrid", hybrid_summary),
            ],
            ignore_index=True,
        )
        st.subheader("Evaluation summary")
        st.dataframe(overview, width="stretch", hide_index=True)
        st.subheader("Per-query assertions")
        st.dataframe(details, width="stretch", hide_index=True)

    with corpus_tab:
        st.subheader("Dataset explorer")
        search_text = st.text_input("Filter documents", value="", key="doc_filter")
        st.dataframe(
            _documents_to_frame(corpus, search_text),
            width="stretch",
            hide_index=True,
        )
        st.subheader("Labeled validation queries")
        st.dataframe(
            _query_cases_to_frame(corpus),
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()
