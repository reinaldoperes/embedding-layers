"""Tests for vector projection helpers."""

import numpy as np

from embedding_layers.corpus import SOCIAL_CORPUS
from embedding_layers.projection import build_projection_frame, project_vectors_to_2d
from embedding_layers.ranking import rank_documents


def test_project_vectors_to_2d_returns_two_columns() -> None:
    """Projection should always return x/y coordinates."""

    vectors = np.array([[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.0, 1.0, 0.0]])

    projected = project_vectors_to_2d(vectors)

    assert projected.shape == (3, 2)


def test_build_projection_frame_includes_query_and_top_documents() -> None:
    """Projection frame should label the query and ranked documents."""

    documents = SOCIAL_CORPUS.documents[:2]
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    ranked_results = rank_documents(documents, [0.9, 0.1])

    frame = build_projection_frame("coffee", documents, vectors, ranked_results)

    assert frame.iloc[0]["kind"] == "query"
    assert set(frame["kind"]) == {"query", "document"}
