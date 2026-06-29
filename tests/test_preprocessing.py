"""Tests for query preprocessing utilities."""

from embedding_layers.corpus import SOCIAL_CORPUS
from embedding_layers.lexical import retrieve_lexical
from embedding_layers.preprocessing import correct_query_with_corpus_vocabulary


def test_corpus_spell_correction_uses_dataset_vocabulary() -> None:
    """Spell correction should map a close typo to a corpus token."""

    corrected_query = correct_query_with_corpus_vocabulary(
        "coffe",
        SOCIAL_CORPUS.documents,
    )

    assert corrected_query == "coffee"


def test_spell_correction_enables_word_lexical_match() -> None:
    """Corrected typos should work with word-based lexical strategies."""

    corrected_query = correct_query_with_corpus_vocabulary(
        "coffe",
        SOCIAL_CORPUS.documents,
    )
    results = retrieve_lexical(
        query=corrected_query,
        documents=SOCIAL_CORPUS.documents,
        strategy_name="TF-IDF word unigrams",
        distance_name="Cosine similarity",
    )
    score_by_id = {result.document.identifier: result.score for result in results}

    assert score_by_id["social-003"] > 0.0
