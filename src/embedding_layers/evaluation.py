"""Retrieval evaluation metrics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from embedding_layers.corpus import Corpus, QueryCase
from embedding_layers.ranking import RankedResult

RetrievalFunction = Callable[[str], list[RankedResult]]


@dataclass(frozen=True)
class QueryEvaluation:
    """Represent validation metrics for one query."""

    query: str
    precision_at_n: float
    average_precision: float


@dataclass(frozen=True)
class EvaluationSummary:
    """Represent validation metrics for a retrieval strategy."""

    strategy_name: str
    precision_at_n: float
    mean_average_precision: float
    query_evaluations: tuple[QueryEvaluation, ...]


def precision_at_n(
    ranked_document_ids: list[str],
    relevant_document_ids: set[str] | frozenset[str],
    n_value: int,
) -> float:
    """Compute precision at N for a ranked list."""

    if n_value <= 0:
        msg = "n_value must be greater than zero"
        raise ValueError(msg)
    top_n = ranked_document_ids[:n_value]
    if not top_n:
        return 0.0
    hits = sum(document_id in relevant_document_ids for document_id in top_n)
    return hits / n_value


def average_precision(
    ranked_document_ids: list[str],
    relevant_document_ids: set[str] | frozenset[str],
) -> float:
    """Compute average precision for a ranked list."""

    if not relevant_document_ids:
        return 0.0

    precision_values: list[float] = []
    hits = 0
    for index, document_id in enumerate(ranked_document_ids, start=1):
        if document_id in relevant_document_ids:
            hits += 1
            precision_values.append(hits / index)

    return sum(precision_values) / len(relevant_document_ids)


def evaluate_query(
    query_case: QueryCase,
    results: list[RankedResult],
    n_value: int,
) -> QueryEvaluation:
    """Evaluate one query case against ranked results."""

    ranked_document_ids = [result.document.identifier for result in results]
    return QueryEvaluation(
        query=query_case.query,
        precision_at_n=precision_at_n(
            ranked_document_ids,
            query_case.relevant_document_ids,
            n_value,
        ),
        average_precision=average_precision(
            ranked_document_ids,
            query_case.relevant_document_ids,
        ),
    )


def evaluate_strategy(
    corpus: Corpus,
    strategy_name: str,
    retrieval_function: RetrievalFunction,
    n_value: int,
) -> EvaluationSummary:
    """Evaluate a retrieval strategy for every labeled query in a corpus."""

    query_evaluations = tuple(
        evaluate_query(query_case, retrieval_function(query_case.query), n_value)
        for query_case in corpus.query_cases
    )
    if not query_evaluations:
        return EvaluationSummary(strategy_name, 0.0, 0.0, ())

    return EvaluationSummary(
        strategy_name=strategy_name,
        precision_at_n=sum(item.precision_at_n for item in query_evaluations)
        / len(query_evaluations),
        mean_average_precision=sum(item.average_precision for item in query_evaluations)
        / len(query_evaluations),
        query_evaluations=query_evaluations,
    )
