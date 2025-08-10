
from server.utils.functions_metadata import function_schema

from datetime import datetime

# Optional heavy imports guarded to prevent circular/import errors in limited environments
try:
    from server.db import SessionLocal, File as DBFile
    from server.utils import compute_sha256
except Exception:  # pragma: no cover
    SessionLocal = None  # type: ignore
    DBFile = None  # type: ignore
    compute_sha256 = None  # type: ignore

# Folder where on-disk copies are saved

@function_schema(
    name="write_file",
    description="Create or update a user file, storing its content on disk (assets folder) and in the database.",
    required_params=["filename", "content", "user_id"],
    optional_params=["mode"]
)
def write_file(filename: str, content: str, user_id: int, mode: str = "overwrite"):
    """Write content to a file, create if it doesn't exist.

    Args:
        filename (str): Logical filename to create or update for the given user.
        content (str): The content to write.
        mode (str, optional): Either "overwrite" (default) or "append". Determines the write behaviour.
        user_id (int): The owning user's ID.

    Returns:
        dict: Details about the written file. Example::

            {
                "status": "success",
                "filename": "example.txt",
                "mode": "overwrite",
                "bytes_written": 1024
            }
    """

    if mode not in {"overwrite", "append"}:
        raise ValueError("mode must be either 'overwrite' or 'append'")

    bytes_content = content.encode("utf-8")
    bytes_written = len(bytes_content)

    # Store/Update in database (DBFile) only
    if not (SessionLocal and DBFile):
        raise RuntimeError("Database layer unavailable; cannot store files.")

        # Proceed with DB operations
    db = SessionLocal()
    try:
        record = db.query(DBFile).filter(
            DBFile.userId == user_id,
            DBFile.filename == filename
        ).first()
        now = datetime.utcnow()

        if record:
            if mode == "append":
                combined = (record.content or b"") + bytes_content
                record.content = combined
                record.size = len(combined)
            else:  # overwrite
                record.content = bytes_content
                record.size = bytes_written
            record.file_hash = compute_sha256(record.content) if compute_sha256 else None
            record.last_modified = now
        else:
            record = DBFile(
                userId=user_id,
                filename=filename,
                size=bytes_written,
                content=bytes_content,
                content_type="text/plain",
                uploaded_at=now,
                file_hash=compute_sha256(bytes_content) if compute_sha256 else None,
            )
            db.add(record)

        db.commit()
        db.refresh(record)
    finally:
        db.close()

    return {
        "status": "success",
        "fileId": record.fileId,
        "filename": filename,
        "mode": mode,
        "bytes_written": bytes_written,
        "total_size": record.size,
    }
    