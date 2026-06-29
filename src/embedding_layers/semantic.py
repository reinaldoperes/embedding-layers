"""Semantic embedding strategies."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import blake2b

import numpy as np
from numpy.typing import NDArray

from embedding_layers.corpus import Document
from embedding_layers.distances import compute_similarity
from embedding_layers.ranking import RankedResult, rank_documents


@dataclass(frozen=True)
class SemanticStrategy:
    """Describe a semantic embedding backend."""

    name: str
    description: str


@dataclass(frozen=True)
class SemanticBackendStatus:
    """Describe which semantic backend was requested and actually used."""

    requested_strategy: str
    actual_backend: str
    fallback_reason: str | None = None

    @property
    def used_fallback(self) -> bool:
        """Return whether the requested backend fell back to another backend."""

        return self.actual_backend != self.requested_strategy


@dataclass(frozen=True)
class SemanticEncodingResult:
    """Represent semantic vectors with backend execution metadata."""

    embeddings: NDArray[np.float64]
    backend_status: SemanticBackendStatus


@dataclass(frozen=True)
class SemanticRetrievalResult:
    """Represent semantic ranking with vectors and backend metadata."""

    results: list[RankedResult]
    embeddings: NDArray[np.float64]
    backend_status: SemanticBackendStatus


SEMANTIC_STRATEGIES: tuple[SemanticStrategy, ...] = (
    SemanticStrategy(
        "Deterministic hashing fallback",
        "Local deterministic dense vectors for offline demonstrations.",
    ),
    SemanticStrategy(
        "SentenceTransformer all-MiniLM-L6-v2",
        "Dense sentence embeddings from a compact transformer model.",
    ),
)


def _hashing_embedding(text: str, dimensions: int = 384) -> NDArray[np.float64]:
    """Create a deterministic dense vector from token hashes."""

    vector = np.zeros(dimensions, dtype=np.float64)
    for token in text.lower().split():
        digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], byteorder="big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


@lru_cache(maxsize=1)
def _get_sentence_transformer_model():
    """Load and cache the sentence-transformer model."""

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _encode_with_sentence_transformer(texts: list[str]) -> NDArray[np.float64]:
    """Encode texts with a sentence-transformer model."""

    model = _get_sentence_transformer_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float64)


def _encode_with_hashing(texts: list[str]) -> NDArray[np.float64]:
    """Encode texts with deterministic hashing vectors."""

    return np.vstack([_hashing_embedding(text) for text in texts])


def encode_semantic_with_status(
    texts: list[str],
    strategy_name: str,
) -> SemanticEncodingResult:
    """Encode texts and report which semantic backend was actually used."""

    if strategy_name == "Deterministic hashing fallback":
        return SemanticEncodingResult(
            embeddings=_encode_with_hashing(texts),
            backend_status=SemanticBackendStatus(
                requested_strategy=strategy_name,
                actual_backend=strategy_name,
            ),
        )
    if strategy_name == "SentenceTransformer all-MiniLM-L6-v2":
        try:
            return SemanticEncodingResult(
                embeddings=_encode_with_sentence_transformer(texts),
                backend_status=SemanticBackendStatus(
                    requested_strategy=strategy_name,
                    actual_backend=strategy_name,
                ),
            )
        except Exception as exc:
            return SemanticEncodingResult(
                embeddings=_encode_with_hashing(texts),
                backend_status=SemanticBackendStatus(
                    requested_strategy=strategy_name,
                    actual_backend="Deterministic hashing fallback",
                    fallback_reason=f"{exc.__class__.__name__}: {exc}",
                ),
            )
    msg = f"Unsupported semantic strategy: {strategy_name}"
    raise ValueError(msg)


def encode_semantic(texts: list[str], strategy_name: str) -> NDArray[np.float64]:
    """Encode texts using the selected semantic strategy."""

    return encode_semantic_with_status(texts, strategy_name).embeddings


def retrieve_semantic_with_status(
    query: str,
    documents: tuple[Document, ...],
    strategy_name: str,
    distance_name: str,
) -> SemanticRetrievalResult:
    """Retrieve documents and report semantic backend execution metadata."""

    texts = [query, *[document.searchable_text for document in documents]]
    encoding_result = encode_semantic_with_status(texts, strategy_name)
    query_vector = encoding_result.embeddings[:1]
    document_vectors = encoding_result.embeddings[1:]
    scores = compute_similarity(query_vector, document_vectors, distance_name)
    return SemanticRetrievalResult(
        results=rank_documents(documents, scores.flatten().tolist()),
        embeddings=encoding_result.embeddings,
        backend_status=encoding_result.backend_status,
    )


def retrieve_semantic(
    query: str,
    documents: tuple[Document, ...],
    strategy_name: str,
    distance_name: str,
) -> list[RankedResult]:
    """Retrieve documents using semantic embeddings."""

    return retrieve_semantic_with_status(
        query=query,
        documents=documents,
        strategy_name=strategy_name,
        distance_name=distance_name,
    ).results
