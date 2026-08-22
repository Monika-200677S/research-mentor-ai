from app.database import SessionLocal
from app.retrieval import rank_papers_bm25


SESSION_ID = "467c22c0-79c0-4cd9-b224-67adaabf6368"

QUERY = "Machine Learning for Healthcare"


def main():

    db = SessionLocal()

    try:

        results = rank_papers_bm25(
            db=db,
            session_id=SESSION_ID,
            query=QUERY,
            top_k=20
        )

        print()
        print("=" * 80)
        print("BM25 RESULTS")
        print("=" * 80)

        for paper in results:

            print()
            print(f"Rank: {paper['rank']}")
            print(f"Score: {paper['bm25_score']}")
            print(f"Title: {paper['title']}")
            print(f"Year: {paper['year']}")
            print("-" * 80)

        print()
        print(f"Returned {len(results)} papers.")

    finally:
        db.close()


if __name__ == "__main__":
    main()