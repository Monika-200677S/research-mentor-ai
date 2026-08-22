import uuid

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import Base, engine, get_db
from app.models import ResearchSession, Paper
from app.semantic_scholar import search_papers


# Create database tables
Base.metadata.create_all(bind=engine)


app = FastAPI(title="GapReader Backend")


# -----------------------------
# CORS
# -----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Request Model
# -----------------------------

class TopicRequest(BaseModel):
    topic: str


# -----------------------------
# Start Research
# -----------------------------

@app.post("/research/start")
def start_research(
    request: TopicRequest,
    db: Session = Depends(get_db)
):

    topic = request.topic.strip()

    if not topic:
        return {
            "status": "FAILED",
            "error": "Research topic cannot be empty."
        }

    # --------------------------------
    # CHECK EXISTING CACHED SESSION
    # --------------------------------

    existing_session = (
        db.query(ResearchSession)
        .filter(
            ResearchSession.topic == topic,
            ResearchSession.status == "READY"
        )
        .order_by(
            ResearchSession.created_at.desc()
        )
        .first()
    )

    if existing_session:

        papers = (
            db.query(Paper)
            .filter(
                Paper.session_id == existing_session.id
            )
            .all()
        )

        print(
            f"Using cached papers for topic: {topic}"
        )

        return {
            "session_id": existing_session.id,
            "status": "READY",
            "cached": True,
            "paper_count": len(papers),
            "papers": [
                {
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "authors": paper.authors,
                    "year": paper.year,
                    "citation_count": paper.citation_count,
                    "url": paper.url,
                }
                for paper in papers
            ],
        }

    # --------------------------------
    # CREATE NEW SESSION
    # --------------------------------

    session_id = str(uuid.uuid4())

    session = ResearchSession(
        id=session_id,
        topic=topic,
        status="RETRIEVING"
    )

    db.add(session)
    db.commit()

    # --------------------------------
    # RETRIEVE PAPERS
    # --------------------------------

    try:

        raw_papers = search_papers(
            topic,
            limit=30
        )

    except Exception as e:

        session.status = "FAILED"
        session.error_message = str(e)

        db.commit()

        return {
            "session_id": session_id,
            "status": "FAILED",
            "error": str(e),
        }

    # --------------------------------
    # SAVE PAPERS
    # --------------------------------

    saved_papers = []

    for p in raw_papers:

        paper_id = p.get("paperId")

        # Skip papers without a valid ID
        if not paper_id:
            continue

        paper = Paper(
            id=str(uuid.uuid4()),
            session_id=session_id,

            # IMPORTANT:
            # Use Semantic Scholar's paperId.
            semantic_scholar_id=paper_id,

            title=p.get("title") or "Untitled",

            abstract=p.get("abstract"),

            authors=", ".join(
                author.get("name", "")
                for author in (p.get("authors") or [])
            ),

            year=p.get("year"),

            citation_count=p.get(
                "citationCount"
            ),

            url=p.get("url"),
        )

        db.add(paper)

        saved_papers.append({
            "title": paper.title,
            "abstract": paper.abstract,
            "authors": paper.authors,
            "year": paper.year,
            "citation_count": paper.citation_count,
            "url": paper.url,
        })

    # --------------------------------
    # COMPLETE SESSION
    # --------------------------------

    session.status = "READY"

    db.commit()

    return {
        "session_id": session_id,
        "status": "READY",
        "cached": False,
        "paper_count": len(saved_papers),
        "papers": saved_papers,
    }