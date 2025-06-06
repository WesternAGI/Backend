"""Database Models Module for LMS Platform.

This module defines the database models and connection setup for the LMS platform.
It includes the User model and database connection configuration.
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, LargeBinary, UniqueConstraint, SmallInteger, BigInteger
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://lms_user:lms_password@localhost:5432/thoth")

print(f"[DB] Using DATABASE_URL: {DATABASE_URL}")

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

class User(Base):
    """User model representing registered users in the system.
    
    This model stores user credentials and settings. The password is stored
    as a hash, never in plain text.
    """
    __tablename__ = "User"
    userId = Column(Integer, primary_key=True, autoincrement=True, index=True)
    """Unique identifier for the user"""
    username = Column(String, unique=True, index=True, nullable=False)
    """Unique username for authentication"""
    hashed_password = Column(String)
    """Bcrypt hash of the user's password"""
    max_file_size = Column(Integer, default=524288000)  # 500MB default max file size
    """Maximum allowed file size in bytes (default: 500MB)"""
    role = Column(SmallInteger, default=0)  # Added user's role, int2 with default 0
    """User's role (int2 with default 0)"""
    phone_number = Column(BigInteger, nullable=True, unique=True, index=True) # Added phone number
    """User's phone number"""
    
    # Relationships
    files = relationship("File", back_populates="user")
    """Relationship to File objects uploaded by this user"""
    queries = relationship("Query", back_populates="user")
    """Relationship to Query objects created by this user"""
    sessions = relationship("Session", back_populates="user")
    """Relationship to Session objects for this user"""


class File(Base):
    """File model representing files uploaded by users.
    
    Attributes:
        fileId: Unique identifier for the file
        filename: Name of the uploaded file
        userId: Foreign key to the user who uploaded the file
        path: Path where the file is stored on the server (nullable)
        size: Size of the file in bytes
        content: Binary content of the file
        content_type: MIME type of the file
        uploaded_at: Timestamp when the file was uploaded
        user: Relationship to the User who owns this file
    """
    __tablename__ = "File"
    fileId = Column(Integer, primary_key=True, autoincrement=True, index=True)
    """Unique identifier for the file"""
    filename = Column(String, nullable=False)
    """Name of the uploaded file"""
    userId = Column(Integer, ForeignKey("User.userId"), nullable=False)
    """Foreign key to the user who uploaded the file"""
    path = Column(String, nullable=True)  # Now nullable since we store content in DB
    """Path where the file is stored on the server (nullable)"""
    size = Column(Integer, nullable=False)
    """Size of the file in bytes"""
    content = Column(LargeBinary, nullable=True)  # Binary content of the file
    """Binary content of the file"""
    content_type = Column(String(255), nullable=True)  # MIME type
    """MIME type of the file"""
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    """Timestamp when the file was uploaded"""
    
    # Relationships
    user = relationship("User", back_populates="files")
    """Relationship to the User who owns this file"""


class Query(Base):
    """Query model representing AI queries made by users.
    
    Attributes:
        queryId: Unique identifier for the query
        userId: Foreign key to the user who made the query
        chatId: Identifier for grouping related queries into conversations
        query_text: The text of the user's query
        response: The AI response to the query
        created_at: Timestamp when the query was made
        user: Relationship to the User who made this query
    """
    __tablename__ = "Query"
    queryId = Column(Integer, primary_key=True, autoincrement=True, index=True)
    """Unique identifier for the query"""
    userId = Column(Integer, ForeignKey("User.userId"), nullable=False)
    """Foreign key to the user who made the query"""
    chatId = Column(String, nullable=True)
    """Identifier for grouping related queries into conversations"""
    query_text = Column(Text, nullable=False)
    """The text of the user's query"""
    response = Column(Text, nullable=True)
    """The AI response to the query"""
    created_at = Column(DateTime, default=datetime.utcnow)
    """Timestamp when the query was made"""
    
    # Relationships
    user = relationship("User", back_populates="queries")
    """Relationship to the User who made this query"""


class Session(Base):
    """Session model for tracking user login sessions.
    
    Attributes:
        sessionId: Unique identifier for the session
        userId: Foreign key to the user who owns this session
        token: Session token for authentication
        expires_at: Timestamp when the session expires
        user: Relationship to the User who owns this session
    """
    __tablename__ = "Session"
    sessionId = Column(Integer, primary_key=True, autoincrement=True, index=True)
    """Unique identifier for the session"""
    userId = Column(Integer, ForeignKey("User.userId"), nullable=False)
    """Foreign key to the user who owns this session"""
    token = Column(String, nullable=False)
    """Session token for authentication"""
    expires_at = Column(DateTime, nullable=False)
    """Timestamp when the session expires"""
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    """Relationship to the User who owns this session"""





# DO NOT run migrations or create tables at import time in serverless environments!
# Run this manually in a migration script or CLI, not here:
# Base.metadata.create_all(bind=engine)
