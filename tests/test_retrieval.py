"""Tests for lexical, semantic fallback, and hybrid retrieval."""

from embedding_layers.corpus import (
    CLINICAL_CORPUS,
    DISTANCE_METRIC_CORPUS,
    EUCLIDEAN_NOISE_CORPUS,
    SOCIAL_CORPUS,
)
from embedding_layers.hybrid import retrieve_hybrid
from embedding_layers.lexical import retrieve_lexical
from embedding_layers.semantic import retrieve_semantic, retrieve_semantic_with_status


def test_lexical_retrieval_returns_all_documents() -> None:
    """Lexical retrieval should rank every document in the corpus."""

    results = retrieve_lexical(
        query="heart attack",
        documents=CLINICAL_CORPUS.documents,
        strategy_name="TF-IDF word unigrams",
        distance_name="Cosine similarity",
    )

    assert len(results) == len(CLINICAL_CORPUS.documents)
    assert results[0].score >= results[-1].score


def test_lexical_retrieval_uses_document_title() -> None:
    """Lexical retrieval should index titles as searchable document text."""

    results = retrieve_lexical(
        query="remote work",
        documents=SOCIAL_CORPUS.documents,
        strategy_name="TF-IDF word unigrams",
        distance_name="Cosine similarity",
    )
    score_by_id = {result.document.identifier: result.score for result in results}

    assert score_by_id["social-001"] > 0.0


def test_character_ngrams_recover_typo_query() -> None:
    """Character n-grams should match small spelling variations."""

    results = retrieve_lexical(
        query="coffe",
        documents=SOCIAL_CORPUS.documents,
        strategy_name="TF-IDF character n-grams",
        distance_name="Cosine similarity",
    )
    score_by_id = {result.document.identifier: result.score for result in results}

    assert score_by_id["social-003"] > 0.0


def test_bm25_retrieval_returns_all_documents() -> None:
    """BM25 retrieval should rank every document in the corpus."""

    results = retrieve_lexical(
        query="heart attack",
        documents=CLINICAL_CORPUS.documents,
        strategy_name="BM25",
        distance_name="Cosine similarity",
    )

    assert len(results) == len(CLINICAL_CORPUS.documents)
    assert results[0].score >= results[-1].score


def test_euclidean_similarity_penalizes_term_magnitude() -> None:
    """Euclidean similarity should expose count-vector magnitude differences."""

    cosine_results = retrieve_lexical(
        query="coffee",
        documents=DISTANCE_METRIC_CORPUS.documents,
        strategy_name="Count vectorizer",
        distance_name="Cosine similarity",
    )
    euclidean_results = retrieve_lexical(
        query="coffee",
        documents=DISTANCE_METRIC_CORPUS.documents,
        strategy_name="Count vectorizer",
        distance_name="Euclidean similarity",
    )
    cosine_scores = {
        result.document.identifier: result.score for result in cosine_results
    }
    euclidean_scores = {
        result.document.identifier: result.score for result in euclidean_results
    }

    assert cosine_scores["distance-001"] == cosine_scores["distance-002"]
    assert euclidean_scores["distance-001"] > euclidean_scores["distance-002"]


def test_euclidean_similarity_penalizes_extra_noise_dimensions() -> None:
    """Euclidean similarity should favor compact exact count vectors."""

    results = retrieve_lexical(
        query="coffee remote",
        documents=EUCLIDEAN_NOISE_CORPUS.documents,
        strategy_name="Count vectorizer",
        distance_name="Euclidean similarity",
    )
    score_by_id = {result.document.identifier: result.score for result in results}

    assert results[0].document.identifier == "noise-001"
    assert score_by_id["noise-001"] > score_by_id["noise-002"]
    assert score_by_id["noise-001"] > score_by_id["noise-003"]


def test_semantic_hashing_fallback_returns_all_documents() -> None:
    """The offline semantic backend should rank every document."""

    results = retrieve_semantic(
        query="heart attack",
        documents=CLINICAL_CORPUS.documents,
        strategy_name="Deterministic hashing fallback",
        distance_name="Cosine similarity",
    )

    assert len(results) == len(CLINICAL_CORPUS.documents)


def test_hybrid_retrieval_keeps_component_scores() -> None:
    """Hybrid retrieval should expose lexical and semantic component scores."""

    results = retrieve_hybrid(
        query="heart attack",
        documents=CLINICAL_CORPUS.documents,
        lexical_strategy_name="TF-IDF word unigrams",
        semantic_strategy_name="Deterministic hashing fallback",
        distance_name="Cosine similarity",
        lexical_weight=0.5,
    )

    assert results[0].lexical_score is not None
    assert results[0].semantic_score is not None


def test_semantic_retrieval_reports_backend_status() -> None:
    """Semantic retrieval should expose the backend used by the execution."""

    output = retrieve_semantic_with_status(
        query="heart attack",
        documents=CLINICAL_CORPUS.documents,
        strategy_name="Deterministic hashing fallback",
        distance_name="Cosine similarity",
    )

    assert len(output.results) == len(CLINICAL_CORPUS.documents)
    assert output.backend_status.actual_backend == "Deterministic hashing fallback"
    assert not output.backend_status.used_fallback
