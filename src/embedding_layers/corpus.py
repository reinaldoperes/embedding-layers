"""Sample corpora and relevance judgments for the embedding layers PoC."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """Represent a retrievable text document."""

    identifier: str
    title: str
    text: str
    domain: str

    @property
    def searchable_text(self) -> str:
        """Return the text fields used by retrieval strategies."""

        return f"{self.title}. {self.text}"


@dataclass(frozen=True)
class QueryCase:
    """Represent a query with expected relevant documents."""

    query: str
    relevant_document_ids: frozenset[str]


@dataclass(frozen=True)
class Corpus:
    """Represent a corpus and its validation queries."""

    name: str
    description: str
    documents: tuple[Document, ...]
    query_cases: tuple[QueryCase, ...]


CLINICAL_CORPUS = Corpus(
    name="Clinical concepts",
    description=(
        "Short clinical statements designed to expose synonymy and terminology "
        "variation between everyday language and medical language."
    ),
    documents=(
        Document(
            "clinical-001",
            "Acute myocardial infarction",
            "The patient was admitted after symptoms consistent with acute "
            "myocardial infarction and elevated cardiac enzymes.",
            "clinical",
        ),
        Document(
            "clinical-002",
            "Heart attack discharge note",
            "A discharge summary describes recovery after a heart attack and "
            "recommends cardiology follow-up.",
            "clinical",
        ),
        Document(
            "clinical-003",
            "Diabetes medication review",
            "The endocrinology team adjusted insulin therapy for type 2 diabetes.",
            "clinical",
        ),
        Document(
            "clinical-004",
            "High blood pressure counseling",
            "Lifestyle changes were recommended to manage hypertension and reduce "
            "cardiovascular risk.",
            "clinical",
        ),
        Document(
            "clinical-005",
            "Respiratory infection",
            "The patient presented with cough, fever, and signs of a respiratory "
            "tract infection.",
            "clinical",
        ),
        Document(
            "clinical-006",
            "Stroke evaluation",
            "Neurology evaluated sudden speech difficulty and unilateral weakness "
            "after a suspected cerebrovascular accident.",
            "clinical",
        ),
        Document(
            "clinical-007",
            "Kidney function decline",
            "Laboratory results showed elevated creatinine and reduced renal "
            "function during follow-up.",
            "clinical",
        ),
        Document(
            "clinical-008",
            "Asthma rescue inhaler",
            "The care plan included an albuterol inhaler for wheezing and acute "
            "shortness of breath.",
            "clinical",
        ),
        Document(
            "clinical-009",
            "Depression screening",
            "The patient reported low mood, poor sleep, and loss of interest in "
            "daily activities.",
            "clinical",
        ),
        Document(
            "clinical-010",
            "Fracture imaging",
            "Radiography confirmed a wrist fracture after a fall on an outstretched "
            "hand.",
            "clinical",
        ),
    ),
    query_cases=(
        QueryCase("heart attack", frozenset({"clinical-001", "clinical-002"})),
        QueryCase("high blood pressure", frozenset({"clinical-004"})),
        QueryCase("diabetes insulin", frozenset({"clinical-003"})),
        QueryCase("stroke weakness", frozenset({"clinical-006"})),
        QueryCase("kidney renal function", frozenset({"clinical-007"})),
        QueryCase("asthma inhaler", frozenset({"clinical-008"})),
    ),
)

SOCIAL_CORPUS = Corpus(
    name="Social media language",
    description=(
        "Short informal posts with abbreviations, sentiment, and lexical noise."
    ),
    documents=(
        Document(
            "social-001",
            "Remote work burnout",
            "Working from home is convenient, but endless calls are making me feel "
            "burned out.",
            "social",
        ),
        Document(
            "social-002",
            "WFH exhaustion",
            "WFH sounded perfect until video meetings took over my whole day.",
            "social",
        ),
        Document(
            "social-003",
            "Coffee enthusiasm",
            "This new espresso blend is exactly what my morning routine needed.",
            "social",
        ),
        Document(
            "social-004",
            "Product launch",
            "Our team shipped a new analytics dashboard for customer success "
            "managers.",
            "social",
        ),
        Document(
            "social-005",
            "Delayed commute",
            "The train delay turned a short commute into a two-hour wait.",
            "social",
        ),
        Document(
            "social-006",
            "Gym motivation",
            "Finally got back to strength training after months away from the gym.",
            "social",
        ),
        Document(
            "social-007",
            "Streaming recommendation",
            "That documentary series was surprisingly thoughtful and worth the "
            "weekend binge.",
            "social",
        ),
        Document(
            "social-008",
            "Phone battery rant",
            "My phone battery drops to ten percent before lunch even with light use.",
            "social",
        ),
        Document(
            "social-009",
            "Remote collaboration win",
            "Async updates and shared docs made distributed teamwork smoother this "
            "week.",
            "social",
        ),
        Document(
            "social-010",
            "Food delivery delay",
            "The delivery app promised dinner in twenty minutes and arrived after "
            "an hour.",
            "social",
        ),
    ),
    query_cases=(
        QueryCase(
            "remote work fatigue",
            frozenset({"social-001", "social-002", "social-009"}),
        ),
        QueryCase("morning coffee", frozenset({"social-003"})),
        QueryCase("analytics dashboard launch", frozenset({"social-004"})),
        QueryCase("commute train delay", frozenset({"social-005"})),
        QueryCase("phone battery problem", frozenset({"social-008"})),
        QueryCase("delivery app late", frozenset({"social-010"})),
    ),
)

ACADEMIC_CORPUS = Corpus(
    name="Academic abstracts",
    description=(
        "Compact academic-style excerpts about retrieval, embeddings, and "
        "evaluation."
    ),
    documents=(
        Document(
            "academic-001",
            "Dense retrieval",
            "Dense retrieval systems encode queries and documents into continuous "
            "vectors for semantic matching.",
            "academic",
        ),
        Document(
            "academic-002",
            "Sparse retrieval",
            "Sparse lexical models such as TF-IDF and BM25 rely on explicit term "
            "matching.",
            "academic",
        ),
        Document(
            "academic-003",
            "Ranking evaluation",
            "Mean average precision summarizes the quality of ranked retrieval "
            "results across multiple queries.",
            "academic",
        ),
        Document(
            "academic-004",
            "Topic modeling",
            "Probabilistic topic models discover latent themes in document "
            "collections.",
            "academic",
        ),
        Document(
            "academic-005",
            "Neural translation",
            "Sequence-to-sequence models map sentences from one language to another.",
            "academic",
        ),
        Document(
            "academic-006",
            "Hybrid search",
            "Hybrid search combines sparse lexical evidence with dense semantic "
            "representations to improve retrieval robustness.",
            "academic",
        ),
        Document(
            "academic-007",
            "Cosine similarity",
            "Cosine similarity compares vector orientation and is widely used in "
            "high-dimensional text representations.",
            "academic",
        ),
        Document(
            "academic-008",
            "Euclidean geometry",
            "Euclidean distance measures straight-line separation between points "
            "in a vector space.",
            "academic",
        ),
        Document(
            "academic-009",
            "Clinical embeddings",
            "Domain-specific language models can represent biomedical terminology "
            "more accurately than general-purpose encoders.",
            "academic",
        ),
        Document(
            "academic-010",
            "Relevance judgments",
            "Information retrieval experiments depend on labeled query-document "
            "pairs that define relevance.",
            "academic",
        ),
    ),
    query_cases=(
        QueryCase("semantic vector search", frozenset({"academic-001"})),
        QueryCase("tfidf lexical matching", frozenset({"academic-002"})),
        QueryCase("mean average precision", frozenset({"academic-003"})),
        QueryCase("hybrid sparse dense search", frozenset({"academic-006"})),
        QueryCase("cosine vector orientation", frozenset({"academic-007"})),
        QueryCase("relevance judgments", frozenset({"academic-010"})),
    ),
)


DISTANCE_METRIC_CORPUS = Corpus(
    name="Distance metric laboratory",
    description=(
        "Controlled corpus for comparing cosine similarity and Euclidean "
        "similarity. It intentionally includes documents with the same lexical "
        "direction but different term magnitudes."
    ),
    documents=(
        Document(
            "distance-001",
            "Coffee",
            "coffee",
            "distance-lab",
        ),
        Document(
            "distance-002",
            "Coffee coffee coffee",
            "coffee coffee coffee coffee coffee coffee",
            "distance-lab",
        ),
        Document(
            "distance-003",
            "Remote",
            "remote",
            "distance-lab",
        ),
        Document(
            "distance-004",
            "Remote remote remote",
            "remote remote remote remote remote remote",
            "distance-lab",
        ),
        Document(
            "distance-005",
            "Mixed coffee remote",
            "coffee remote coffee remote",
            "distance-lab",
        ),
        Document(
            "distance-006",
            "Unrelated analytics",
            "dashboard metrics analytics report",
            "distance-lab",
        ),
    ),
    query_cases=(
        QueryCase("coffee", frozenset({"distance-001", "distance-002"})),
        QueryCase("remote", frozenset({"distance-003", "distance-004"})),
        QueryCase("coffee remote", frozenset({"distance-005"})),
    ),
)


EUCLIDEAN_NOISE_CORPUS = Corpus(
    name="Euclidean noise laboratory",
    description=(
        "Controlled corpus for observing how Euclidean distance reacts when "
        "documents contain the query terms plus additional unrelated dimensions."
    ),
    documents=(
        Document(
            "noise-001",
            "Coffee remote",
            "",
            "euclidean-noise-lab",
        ),
        Document(
            "noise-002",
            "Coffee remote",
            "dashboard metrics analytics planning budget timeline",
            "euclidean-noise-lab",
        ),
        Document(
            "noise-003",
            "Coffee remote",
            "photos music weekend streaming friends dinner",
            "euclidean-noise-lab",
        ),
        Document(
            "noise-004",
            "Coffee",
            "",
            "euclidean-noise-lab",
        ),
        Document(
            "noise-005",
            "Remote",
            "",
            "euclidean-noise-lab",
        ),
        Document(
            "noise-006",
            "Unrelated project metrics",
            "dashboard metrics analytics report planning budget timeline",
            "euclidean-noise-lab",
        ),
    ),
    query_cases=(
        QueryCase("coffee remote", frozenset({"noise-001"})),
        QueryCase("coffee", frozenset({"noise-004"})),
        QueryCase("remote", frozenset({"noise-005"})),
    ),
)

CORPORA: tuple[Corpus, ...] = (
    CLINICAL_CORPUS,
    SOCIAL_CORPUS,
    ACADEMIC_CORPUS,
    DISTANCE_METRIC_CORPUS,
    EUCLIDEAN_NOISE_CORPUS,
)


def get_corpus_by_name(name: str) -> Corpus:
    """Return a corpus by its display name."""

    for corpus in CORPORA:
        if corpus.name == name:
            return corpus
    msg = f"Unknown corpus: {name}"
    raise ValueError(msg)
