from typing import List, Dict

from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session

from app.models import Paper
from app.preprocessing import preprocess_text


def build_document_text(paper: Paper) -> str:
    """
    Combine the paper title and abstract.

    The title and abstract together provide the text
    used for BM25 relevance ranking.
    """

    title = paper.title or ""
    abstract = paper.abstract or ""

    return f"{title} {abstract}"


def rank_papers_bm25(
    db: Session,
    session_id: str,
    query: str,
    top_k: int = 50
) -> List[Dict]:
    """
    Rank papers using BM25 after classical NLP preprocessing.
    """

    # -----------------------------------------
    # 1. Get papers from SQLite
    # -----------------------------------------

    papers = (
        db.query(Paper)
        .filter(Paper.session_id == session_id)
        .all()
    )

    if not papers:
        return []

    # -----------------------------------------
    # 2. Preprocess every paper
    # -----------------------------------------

    documents = []
    valid_papers = []

    for paper in papers:

        text = build_document_text(paper)

        tokens = preprocess_text(text)

        if not tokens:
            continue

        documents.append(tokens)
        valid_papers.append(paper)

    if not documents:
        return []

    # -----------------------------------------
    # 3. Build BM25 index
    # -----------------------------------------

    bm25 = BM25Okapi(
        documents,
        k1=1.5,
        b=0.75
    )

    # -----------------------------------------
    # 4. Preprocess user query
    # -----------------------------------------

    query_tokens = preprocess_text(query)

    if not query_tokens:
        return []

    # -----------------------------------------
    # 5. Calculate BM25 scores
    # -----------------------------------------

    scores = bm25.get_scores(query_tokens)

    # -----------------------------------------
    # 6. Sort by score
    # -----------------------------------------

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    # -----------------------------------------
    # 7. Return top K
    # -----------------------------------------

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
            "bm25_score": round(
                float(scores[index]),
                4
            )
        })

    return results