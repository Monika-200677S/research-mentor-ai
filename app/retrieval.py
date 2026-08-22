from typing import List, Dict

from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session

from app.models import Paper
from app.preprocessing import preprocess_text


# --------------------------------------------------
# BM25 configuration
# --------------------------------------------------

# Give the title more importance because it directly
# represents the main subject of the paper.
TITLE_WEIGHT = 2.0

# The abstract provides additional contextual information.
ABSTRACT_WEIGHT = 1.0


# Standard BM25 parameters.
BM25_K1 = 1.5
BM25_B = 0.75


def build_bm25_index(texts: List[List[str]]) -> BM25Okapi:
    """
    Build a BM25 index from preprocessed documents.
    """

    return BM25Okapi(
        texts,
        k1=BM25_K1,
        b=BM25_B
    )


def rank_papers_bm25(
    db: Session,
    session_id: str,
    query: str,
    top_k: int = 50
) -> List[Dict]:
    """
    Rank papers using field-weighted BM25.

    Title and abstract are treated as separate fields.

    Final score:

        (normalized title BM25 × TITLE_WEIGHT)
        +
        (normalized abstract BM25 × ABSTRACT_WEIGHT)
    """

    # --------------------------------------------------
    # 1. Retrieve papers from SQLite
    # --------------------------------------------------

    papers = (
        db.query(Paper)
        .filter(Paper.session_id == session_id)
        .all()
    )

    if not papers:
        return []

    # --------------------------------------------------
    # 2. Prepare title and abstract separately
    # --------------------------------------------------

    valid_papers = []

    title_documents = []
    abstract_documents = []

    for paper in papers:

        title_tokens = preprocess_text(
            paper.title or ""
        )

        abstract_tokens = preprocess_text(
            paper.abstract or ""
        )

        # Skip papers that contain no usable text.
        if not title_tokens and not abstract_tokens:
            continue

        valid_papers.append(paper)

        title_documents.append(title_tokens)
        abstract_documents.append(abstract_tokens)

    if not valid_papers:
        return []

    # --------------------------------------------------
    # 3. Build separate BM25 indexes
    # --------------------------------------------------

    title_bm25 = build_bm25_index(
        title_documents
    )

    abstract_bm25 = build_bm25_index(
        abstract_documents
    )

    # --------------------------------------------------
    # 4. Preprocess the user's query
    # --------------------------------------------------

    query_tokens = preprocess_text(query)

    if not query_tokens:
        return []

    # --------------------------------------------------
    # 5. Calculate BM25 scores for each field
    # --------------------------------------------------

    title_scores = title_bm25.get_scores(
        query_tokens
    )

    abstract_scores = abstract_bm25.get_scores(
        query_tokens
    )

    # --------------------------------------------------
    # 6. Normalize title and abstract scores
    # --------------------------------------------------

    max_title_score = max(title_scores)
    max_abstract_score = max(abstract_scores)

    if max_title_score > 0:
        normalized_title_scores = (
            title_scores / max_title_score
        )
    else:
        normalized_title_scores = title_scores

    if max_abstract_score > 0:
        normalized_abstract_scores = (
            abstract_scores / max_abstract_score
        )
    else:
        normalized_abstract_scores = abstract_scores

    # --------------------------------------------------
    # 7. Calculate final weighted score
    # --------------------------------------------------

    final_scores = (
        TITLE_WEIGHT * normalized_title_scores
        +
        ABSTRACT_WEIGHT * normalized_abstract_scores
    )

    # --------------------------------------------------
    # 8. Rank papers by final score
    # --------------------------------------------------

    ranked_indices = sorted(
        range(len(final_scores)),
        key=lambda i: final_scores[i],
        reverse=True
    )

    # --------------------------------------------------
    # 9. Return top K papers
    # --------------------------------------------------

    results = []

    for rank, index in enumerate(
        ranked_indices[:top_k],
        start=1
    ):

        paper = valid_papers[index]

        results.append({
            "rank": rank,
            "paper_id": paper.id,
            "semantic_scholar_id": paper.semantic_scholar_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "authors": paper.authors,
            "year": paper.year,
            "citation_count": paper.citation_count,
            "url": paper.url,

            # Normalized title contribution
            "title_bm25_score": round(
                float(normalized_title_scores[index]),
                4
            ),

            # Normalized abstract contribution
            "abstract_bm25_score": round(
                float(normalized_abstract_scores[index]),
                4
            ),

            # Final weighted BM25 score
            "bm25_score": round(
                float(final_scores[index]),
                4
            )
        })

    return results