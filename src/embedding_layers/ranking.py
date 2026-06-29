"""Ranking primitives shared by retrieval strategies."""

from __future__ import annotations

from dataclasses import dataclass

from embedding_layers.corpus import Document


@dataclass(frozen=True)
class RankedResult:
    """Represent one ranked document result."""

    rank: int
    document: Document
    score: float
    lexical_score: float | None = None
    semantic_score: float | None = None


def rank_documents(
    documents: tuple[Document, ...],
    scores: list[float],
    lexical_scores: list[float] | None = None,
    semantic_scores: list[float] | None = None,
) -> list[RankedResult]:
    """Create ranked results from score arrays."""

    ordered_indexes = sorted(
        range(len(documents)),
        key=lambda index: scores[index],
        reverse=True,
    )
    results: list[RankedResult] = []
    for position, index in enumerate(ordered_indexes, start=1):
        results.append(
            RankedResult(
                rank=position,
                document=documents[index],
                score=float(scores[index]),
                lexical_score=(
                    None if lexical_scores is None else float(lexical_scores[index])
                ),
                semantic_score=(
                    None if semantic_scores is None else float(semantic_scores[index])
                ),
            )
        )
    return results
