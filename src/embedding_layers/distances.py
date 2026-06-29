"""Distance and similarity functions for vector retrieval."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

VectorMatrix = NDArray[np.float64]


def cosine_similarity(
    query_vector: VectorMatrix,
    document_vectors: VectorMatrix,
) -> VectorMatrix:
    """Compute cosine similarity between one query vector and document vectors."""

    query_norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
    document_norms = np.linalg.norm(document_vectors, axis=1, keepdims=True)
    denominator = np.maximum(query_norm * document_norms.T, 1e-12)
    return (query_vector @ document_vectors.T) / denominator


def euclidean_similarity(
    query_vector: VectorMatrix,
    document_vectors: VectorMatrix,
) -> VectorMatrix:
    """Convert Euclidean distance into a bounded similarity score."""

    distances = np.linalg.norm(document_vectors - query_vector, axis=1)
    return (1.0 / (1.0 + distances)).reshape(1, -1)


def compute_similarity(
    query_vector: VectorMatrix,
    document_vectors: VectorMatrix,
    distance_name: str,
) -> VectorMatrix:
    """Compute similarity using the selected distance strategy."""

    if distance_name == "Cosine similarity":
        return cosine_similarity(query_vector, document_vectors)
    if distance_name == "Euclidean similarity":
        return euclidean_similarity(query_vector, document_vectors)
    msg = f"Unsupported distance strategy: {distance_name}"
    raise ValueError(msg)
