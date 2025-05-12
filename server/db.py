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
    
    Attributes:
        userId: Unique identifier for the user
        username: Unique username for authentication
        hashed_password: Bcrypt hash of the user's password
        max_file_size: Maximum allowed file size in bytes (default: 500MB)
        role: User's role (int2 with default 0)
        phone_number: User's phone number
        files: Relationship to File objects uploaded by this user
        queries: Relationship to Query objects created by this user
        sessions: Relationship to Session objects for this user
    """
    __tablename__ = "User"
    userId = Column(Integer, primary_key=True, autoincrement=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String)
    max_file_size = Column(Integer, default=524288000)  # 500MB default max file size
    role = Column(SmallInteger, default=0)  # Added user's role, int2 with default 0
    phone_number = Column(BigInteger, nullable=True, unique=True, index=True) # Added phone number
    
    # Relationships
    files = relationship("File", back_populates="user")
    queries = relationship("Query", back_populates="user")
    sessions = relationship("Session", back_populates="user")


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
        versions: Relationship to FileVersion objects for this file
        metadata: Relationship to FileMetadata objects for this file
    """
    __tablename__ = "File"
    fileId = Column(Integer, primary_key=True, autoincrement=True, index=True)
    filename = Column(String, nullable=False)
    userId = Column(Integer, ForeignKey("User.userId"), nullable=False)
    path = Column(String, nullable=True)  # Now nullable since we store content in DB
    size = Column(Integer, nullable=False)
    content = Column(LargeBinary, nullable=True)  # Binary content of the file
    content_type = Column(String(255), nullable=True)  # MIME type
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="files")
    versions = relationship("FileVersion", back_populates="file", cascade="all, delete-orphan")
    file_metadata = relationship("FileMetadata", back_populates="file", cascade="all, delete-orphan")


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
    userId = Column(Integer, ForeignKey("User.userId"), nullable=False)
    chatId = Column(String, nullable=True)
    query_text = Column(Text, nullable=False)
    response = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="queries")


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
    userId = Column(Integer, ForeignKey("User.userId"), nullable=False)
    token = Column(String, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="sessions")

class FileVersion(Base):
    """FileVersion model for tracking different versions of files.
    
    Attributes:
        versionId: Unique identifier for the version
        fileId: Foreign key to the file this version belongs to
        userId: Foreign key to the user who created this version
        content: Binary content of this version
        size: Size of the file version in bytes
        version_number: Sequential version number for this file
        created_at: Timestamp when the version was created
        file: Relationship to the File this version belongs to
    """
    __tablename__ = "FileVersion"
    versionId = Column(Integer, primary_key=True, autoincrement=True, index=True)
    fileId = Column(Integer, ForeignKey("File.fileId"), nullable=False)
    userId = Column(Integer, ForeignKey("User.userId"), nullable=False)
    content = Column(LargeBinary, nullable=True)
    size = Column(Integer, nullable=False)
    version_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    file = relationship("File", back_populates="versions")
    
    # Ensure each file can only have one version with a specific number
    __table_args__ = (UniqueConstraint('fileId', 'version_number', name='_file_version_uc'),)


class FileMetadata(Base):
    """FileMetadata model for storing additional file properties.
    
    Attributes:
        metadataId: Unique identifier for the metadata entry
        fileId: Foreign key to the file this metadata belongs to
        key: Name of the metadata property
        value: Value of the metadata property
        created_at: Timestamp when the metadata was created
        file: Relationship to the File this metadata belongs to
    """
    __tablename__ = "FileMetadata"
    metadataId = Column(Integer, primary_key=True, autoincrement=True, index=True)
    fileId = Column(Integer, ForeignKey("File.fileId"), nullable=False)
    key = Column(String(255), nullable=False)
    value = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    file = relationship("File", back_populates="file_metadata")
    
    # Ensure each file can only have one entry for a specific key
    __table_args__ = (UniqueConstraint('fileId', 'key', name='_file_metadata_key_uc'),)


# DO NOT run migrations or create tables at import time in serverless environments!
# Run this manually in a migration script or CLI, not here:
# Base.metadata.create_all(bind=engine)
