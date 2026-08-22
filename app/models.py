from sqlalchemy import Column, String, Integer, Text, Boolean, ForeignKey, DateTime
from sqlalchemy.sql import func
from app.database import Base


class ResearchSession(Base):
    __tablename__ = "research_sessions"

    id = Column(String, primary_key=True)
    topic = Column(String, nullable=False)
    status = Column(String, default="PENDING")  # PENDING, RETRIEVING, READY, FAILED
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Paper(Base):
    __tablename__ = "papers"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("research_sessions.id"))
    semantic_scholar_id = Column(String, nullable=True)
    title = Column(Text, nullable=False)
    abstract = Column(Text, nullable=True)
    authors = Column(Text, nullable=True)     # stored as comma-separated string for now
    year = Column(Integer, nullable=True)
    citation_count = Column(Integer, nullable=True)
    url = Column(Text, nullable=True)