"""Optional external dataset loaders for the Streamlit PoC."""

from __future__ import annotations

from collections.abc import Iterable

from embedding_layers.corpus import Corpus, Document, QueryCase

NEWSGROUP_CATEGORIES = ("sci.space", "rec.autos", "comp.graphics")
AG_NEWS_DATASET_CANDIDATES = ("fancyzhx/ag_news", "ag_news")
AG_NEWS_LABELS = {
    0: "world",
    1: "sports",
    2: "business",
    3: "sci_tech",
}


def _clean_text(text: str, max_chars: int = 900) -> str:
    """Normalize whitespace and limit text length for interactive demos."""

    return " ".join(text.split())[:max_chars]


def _relevance_by_domain(
    documents: Iterable[Document],
    domain: str,
) -> frozenset[str]:
    """Return document identifiers for one domain label."""

    return frozenset(
        document.identifier for document in documents if document.domain == domain
    )


def load_20newsgroups_sample(documents_per_category: int = 12) -> Corpus:
    """Download or load a compact 20 Newsgroups retrieval corpus.

    Scikit-learn downloads the dataset on first use and caches it locally. The
    resulting corpus is intentionally small so the Streamlit PoC remains fast.
    """

    from sklearn.datasets import fetch_20newsgroups

    dataset = fetch_20newsgroups(
        subset="train",
        categories=list(NEWSGROUP_CATEGORIES),
        remove=("headers", "footers", "quotes"),
    )
    category_counts = dict.fromkeys(NEWSGROUP_CATEGORIES, 0)
    documents: list[Document] = []
    for index, text in enumerate(dataset.data):
        category = dataset.target_names[dataset.target[index]]
        if category_counts[category] >= documents_per_category:
            continue
        category_counts[category] += 1
        documents.append(
            Document(
                identifier=f"20ng-{category}-{category_counts[category]:03d}",
                title=f"20 Newsgroups: {category} #{category_counts[category]}",
                text=_clean_text(text),
                domain=category,
            )
        )
        if all(count >= documents_per_category for count in category_counts.values()):
            break

    query_cases = (
        QueryCase("space nasa orbit", _relevance_by_domain(documents, "sci.space")),
        QueryCase(
            "car engine automotive",
            _relevance_by_domain(documents, "rec.autos"),
        ),
        QueryCase(
            "computer graphics image rendering",
            _relevance_by_domain(documents, "comp.graphics"),
        ),
    )
    return Corpus(
        name="20 Newsgroups sample",
        description=(
            "Downloaded sample from scikit-learn's 20 Newsgroups dataset. "
            "Labels are category-derived, so they are useful for exploration but "
            "less precise than the hand-labeled teaching corpora."
        ),
        documents=tuple(documents),
        query_cases=query_cases,
    )


def _load_huggingface_dataset(dataset_names: str | tuple[str, ...], split: str):
    """Load a Hugging Face dataset with fallback dataset identifiers."""

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        msg = (
            "The optional Hugging Face datasets dependency is not installed. "
            "Install it with: pip install -e '.[datasets]'"
        )
        raise RuntimeError(msg) from exc

    candidates = (dataset_names,) if isinstance(dataset_names, str) else dataset_names
    errors: list[str] = []
    for dataset_name in candidates:
        try:
            return load_dataset(dataset_name, split=split)
        except Exception as exc:
            errors.append(f"{dataset_name}: {exc}")

    msg = "All Hugging Face dataset candidates failed. " + " | ".join(errors)
    raise RuntimeError(msg)


def load_ag_news_cosine_sample(documents_per_label: int = 12) -> Corpus:
    """Download AG News for topical cosine and semantic retrieval experiments."""

    dataset = _load_huggingface_dataset(AG_NEWS_DATASET_CANDIDATES, "train")
    label_counts = dict.fromkeys(AG_NEWS_LABELS.values(), 0)
    documents: list[Document] = []
    for row in dataset:
        domain = AG_NEWS_LABELS[int(row["label"])]
        if label_counts[domain] >= documents_per_label:
            continue
        label_counts[domain] += 1
        documents.append(
            Document(
                identifier=f"ag-news-{domain}-{label_counts[domain]:03d}",
                title=f"AG News: {domain} #{label_counts[domain]}",
                text=_clean_text(row["text"]),
                domain=domain,
            )
        )
        if all(count >= documents_per_label for count in label_counts.values()):
            break

    query_cases = (
        QueryCase(
            "global politics government",
            _relevance_by_domain(documents, "world"),
        ),
        QueryCase(
            "team game season",
            _relevance_by_domain(documents, "sports"),
        ),
        QueryCase(
            "market company profit",
            _relevance_by_domain(documents, "business"),
        ),
        QueryCase(
            "technology software internet",
            _relevance_by_domain(documents, "sci_tech"),
        ),
    )
    return Corpus(
        name="AG News topical sample",
        description=(
            "Downloaded AG News sample from Hugging Face datasets. This corpus is "
            "useful for cosine and semantic experiments over topical news text."
        ),
        documents=tuple(documents),
        query_cases=query_cases,
    )


def load_ag_news_euclidean_length_sample(documents_per_label: int = 10) -> Corpus:
    """Download AG News for Euclidean experiments with real length variation."""

    dataset = _load_huggingface_dataset(AG_NEWS_DATASET_CANDIDATES, "train")
    label_counts = dict.fromkeys(AG_NEWS_LABELS.values(), 0)
    documents: list[Document] = []
    for row in dataset:
        domain = AG_NEWS_LABELS[int(row["label"])]
        if label_counts[domain] >= documents_per_label:
            continue
        label_counts[domain] += 1
        text = _clean_text(row["text"], max_chars=1400)
        token_count = len(text.split())
        documents.append(
            Document(
                identifier=f"ag-euclidean-{domain}-{label_counts[domain]:03d}",
                title=(
                    f"AG Euclidean: {domain} #{label_counts[domain]} "
                    f"({token_count} tokens)"
                ),
                text=text,
                domain=domain,
            )
        )
        if all(count >= documents_per_label for count in label_counts.values()):
            break

    query_cases = (
        QueryCase(
            "technology software internet",
            _relevance_by_domain(documents, "sci_tech"),
        ),
        QueryCase(
            "market company profit",
            _relevance_by_domain(documents, "business"),
        ),
        QueryCase(
            "team game season",
            _relevance_by_domain(documents, "sports"),
        ),
    )
    return Corpus(
        name="AG News Euclidean length sample",
        description=(
            "Downloaded AG News sample for Euclidean distance experiments. It uses "
            "real news snippets with different lengths, making Count vectorizer + "
            "Euclidean similarity visibly sensitive to extra terms and magnitude."
        ),
        documents=tuple(documents),
        query_cases=query_cases,
    )
