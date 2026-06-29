"""Tests for retrieval evaluation metrics."""

from embedding_layers.evaluation import average_precision, precision_at_n


def test_precision_at_n_counts_relevant_documents_in_top_n() -> None:
    """P@N should count relevant results only inside the prefix."""

    ranked_ids = ["a", "b", "c", "d"]
    relevant_ids = {"b", "d"}

    assert precision_at_n(ranked_ids, relevant_ids, 2) == 0.5


def test_average_precision_uses_precision_at_relevant_ranks() -> None:
    """Average precision should average precision only at relevant ranks."""

    ranked_ids = ["a", "b", "c", "d"]
    relevant_ids = {"b", "d"}

    assert average_precision(ranked_ids, relevant_ids) == 0.5
