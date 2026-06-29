"""Hybrid retrieval that combines lexical and semantic scores."""

from __future__ import annotations

from embedding_layers.corpus import Document
from embedding_layers.lexical import retrieve_lexical
from embedding_layers.ranking import RankedResult, rank_documents
from embedding_layers.semantic import retrieve_semantic


def _scores_by_document_id(results: list[RankedResult]) -> dict[str, float]:
    """Index ranking scores by document identifier."""

    return {result.document.identifier: result.score for result in results}


def retrieve_hybrid(
    query: str,
    documents: tuple[Document, ...],
    lexical_strategy_name: str,
    semantic_strategy_name: str,
    distance_name: str,
    lexical_weight: float,
) -> list[RankedResult]:
    """Retrieve documents by combining lexical and semantic scores."""

    if not 0.0 <= lexical_weight <= 1.0:
        msg = "lexical_weight must be between 0.0 and 1.0"
        raise ValueError(msg)

    lexical_results = retrieve_lexical(
        query=query,
        documents=documents,
        strategy_name=lexical_strategy_name,
        distance_name=distance_name,
    )
    semantic_results = retrieve_semantic(
        query=query,
        documents=documents,
        strategy_name=semantic_strategy_name,
        distance_name=distance_name,
    )
    lexical_scores = _scores_by_document_id(lexical_results)
    semantic_scores = _scores_by_document_id(semantic_results)

    combined_scores: list[float] = []
    ordered_lexical_scores: list[float] = []
    ordered_semantic_scores: list[float] = []
    for document in documents:
        lexical_score = lexical_scores[document.identifier]
        semantic_score = semantic_scores[document.identifier]
        combined_scores.append(
            lexical_weight * lexical_score + (1.0 - lexical_weight) * semantic_score
        )
        ordered_lexical_scores.append(lexical_score)
        ordered_semantic_scores.append(semantic_score)

    return rank_documents(
        documents=documents,
        scores=combined_scores,
        lexical_scores=ordered_lexical_scores,
        semantic_scores=ordered_semantic_scores,
    )
