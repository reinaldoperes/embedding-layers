"""Vector projection helpers for visual retrieval explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from embedding_layers.corpus import Document
from embedding_layers.ranking import RankedResult


def project_vectors_to_2d(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project vectors to two dimensions for explanatory visualization."""

    if vectors.shape[0] < 2:
        return np.zeros((vectors.shape[0], 2), dtype=np.float64)

    centered_vectors = vectors - vectors.mean(axis=0, keepdims=True)
    if np.allclose(centered_vectors, 0.0):
        return np.zeros((vectors.shape[0], 2), dtype=np.float64)

    component_count = min(2, vectors.shape[0], vectors.shape[1])
    projected = PCA(n_components=component_count).fit_transform(centered_vectors)
    if component_count == 1:
        return np.column_stack([projected[:, 0], np.zeros(vectors.shape[0])])
    return projected.astype(np.float64)


def build_projection_frame(
    query: str,
    documents: tuple[Document, ...],
    vectors: NDArray[np.float64],
    ranked_results: list[RankedResult],
    max_documents: int = 12,
) -> pd.DataFrame:
    """Build a dataframe with projected query and top-ranked document vectors."""

    coordinates = project_vectors_to_2d(vectors)
    score_by_id = {
        result.document.identifier: result.score for result in ranked_results
    }
    rank_by_id = {result.document.identifier: result.rank for result in ranked_results}
    top_document_ids = {
        result.document.identifier for result in ranked_results[:max_documents]
    }
    rows = [
        {
            "label": f"Query: {query}",
            "kind": "query",
            "x": coordinates[0, 0],
            "y": coordinates[0, 1],
            "score": 1.0,
            "size": 180,
        }
    ]
    for index, document in enumerate(documents, start=1):
        if document.identifier not in top_document_ids:
            continue
        rows.append(
            {
                "label": f"#{rank_by_id[document.identifier]} {document.title}",
                "kind": "document",
                "x": coordinates[index, 0],
                "y": coordinates[index, 1],
                "score": round(score_by_id[document.identifier], 4),
                "size": 70,
            }
        )
    return pd.DataFrame(rows)
