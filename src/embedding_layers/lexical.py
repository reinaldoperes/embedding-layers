"""Lexical vectorization strategies."""

from __future__ import annotations

from dataclasses import dataclass
from math import log

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from embedding_layers.corpus import Document
from embedding_layers.distances import compute_similarity
from embedding_layers.ranking import RankedResult, rank_documents


@dataclass(frozen=True)
class LexicalStrategy:
    """Describe a lexical vectorization strategy."""

    name: str
    description: str


LEXICAL_STRATEGIES: tuple[LexicalStrategy, ...] = (
    LexicalStrategy(
        "Count vectorizer",
        "Raw token frequency. Useful as the most transparent lexical baseline.",
    ),
    LexicalStrategy(
        "TF-IDF word unigrams",
        "Term frequency weighted by inverse document frequency using single words.",
    ),
    LexicalStrategy(
        "TF-IDF word unigrams+bigrams",
        "TF-IDF using single words and two-word phrases.",
    ),
    LexicalStrategy(
        "TF-IDF character n-grams",
        "TF-IDF over character fragments, useful for noisy or "
        "morphologically related text.",
    ),
    LexicalStrategy(
        "BM25",
        "Probabilistic lexical ranking with term saturation and length normalization.",
    ),
)


def _build_vectorizer(strategy_name: str) -> CountVectorizer | TfidfVectorizer:
    """Create a scikit-learn vectorizer for a lexical strategy."""

    if strategy_name == "Count vectorizer":
        return CountVectorizer(stop_words="english")
    if strategy_name == "TF-IDF word unigrams":
        return TfidfVectorizer(stop_words="english", ngram_range=(1, 1))
    if strategy_name == "TF-IDF word unigrams+bigrams":
        return TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    if strategy_name == "TF-IDF character n-grams":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    msg = f"Unsupported lexical strategy: {strategy_name}"
    raise ValueError(msg)


def _tokenize(text: str) -> list[str]:
    """Tokenize text for the lightweight BM25 implementation."""

    return [token.strip(".,;:!?()[]{}\"'").lower() for token in text.split()]


def _retrieve_bm25(query: str, documents: tuple[Document, ...]) -> list[RankedResult]:
    """Retrieve documents using a compact BM25 implementation."""

    k1 = 1.5
    b_value = 0.75
    tokenized_documents = [
        _tokenize(document.searchable_text) for document in documents
    ]
    query_terms = _tokenize(query)
    average_length = sum(len(tokens) for tokens in tokenized_documents) / len(
        tokenized_documents
    )
    document_frequency: dict[str, int] = {}
    for tokens in tokenized_documents:
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1

    scores: list[float] = []
    document_count = len(documents)
    for tokens in tokenized_documents:
        term_frequency = {token: tokens.count(token) for token in set(tokens)}
        document_length = len(tokens)
        score = 0.0
        for term in query_terms:
            if term not in term_frequency:
                continue
            inverse_document_frequency = log(
                1.0
                + (document_count - document_frequency.get(term, 0) + 0.5)
                / (document_frequency.get(term, 0) + 0.5)
            )
            numerator = term_frequency[term] * (k1 + 1.0)
            denominator = term_frequency[term] + k1 * (
                1.0 - b_value + b_value * document_length / average_length
            )
            score += inverse_document_frequency * numerator / denominator
        scores.append(score)
    return rank_documents(documents, scores)


def retrieve_lexical(
    query: str,
    documents: tuple[Document, ...],
    strategy_name: str,
    distance_name: str,
) -> list[RankedResult]:
    """Retrieve documents using a lexical vectorization strategy."""

    if strategy_name == "BM25":
        return _retrieve_bm25(query, documents)

    vectorizer = _build_vectorizer(strategy_name)
    document_texts = [document.searchable_text for document in documents]
    document_vectors = vectorizer.fit_transform(document_texts).toarray()
    query_vector = vectorizer.transform([query]).toarray()
    scores = compute_similarity(query_vector, document_vectors, distance_name)
    return rank_documents(documents, scores.flatten().tolist())
