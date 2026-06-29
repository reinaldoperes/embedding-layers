"""Query preprocessing utilities for retrieval experiments."""

from __future__ import annotations

import re
from difflib import get_close_matches

from embedding_layers.corpus import Document

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]*")


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into lowercase lexical units."""

    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def build_corpus_vocabulary(documents: tuple[Document, ...]) -> set[str]:
    """Build a token vocabulary from searchable document text."""

    vocabulary: set[str] = set()
    for document in documents:
        vocabulary.update(tokenize_text(document.searchable_text))
    return vocabulary


def correct_query_with_corpus_vocabulary(
    query: str,
    documents: tuple[Document, ...],
    cutoff: float = 0.82,
) -> str:
    """Correct query tokens using close matches from the selected corpus.

    This intentionally uses the active corpus vocabulary rather than a global
    dictionary. The behavior is transparent for a PoC and avoids introducing an
    external dependency whose language/domain assumptions would be hidden.
    """

    vocabulary = build_corpus_vocabulary(documents)
    corrected_tokens: list[str] = []
    for token in tokenize_text(query):
        if token in vocabulary:
            corrected_tokens.append(token)
            continue
        matches = get_close_matches(token, vocabulary, n=1, cutoff=cutoff)
        corrected_tokens.append(matches[0] if matches else token)
    return " ".join(corrected_tokens)
