"""API Routes Module for AI-Powered Backend Platform.

This module defines all the API endpoints for the platform, including:
- User authentication and management
- File operations (upload, download, listing, deletion)
- AI query handling and conversation management
- Device management and tracking
- Twilio integration for SMS/voice
- User profile and settings

All endpoints are protected with JWT authentication unless explicitly marked as public.
"""

import os
import re
import logging
import json
import uuid
import mimetypes
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from urllib.parse import quote

# Configure logging
logger = logging.getLogger(__name__)

# Local imports
from server.utils.logging_utils import (
    log_request_start, 
    log_response,
    log_error,
    log_something,
    log_ai_call,
    log_ai_response,
    log_request_payload,
    logger
)

# AI Components - using local package
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

# Now import the local aiagent package
try:
    from aiagent.memory.memory_manager import LongTermMemoryManager, ShortTermMemoryManager
    from aiagent.handler.query import query_openai, summarize_conversation, update_memory
    
    # Create a simple wrapper class for the query handler
    class AIQueryHandler:
        def __init__(self):
            self.query_openai = query_openai
            self.summarize_conversation = summarize_conversation
            self.update_memory = update_memory
    
    ai_query_handler = AIQueryHandler()
    logger.info("AI Query Handler initialized successfully")
    
except ImportError as e:
    logger.error(f"Failed to import AI components: {str(e)}")
    # Create a dummy handler to prevent errors
    class DummyAIHandler:
        def query_openai(self, *args, **kwargs):
            return "AI functionality is currently unavailable. Please check the server logs."
        def summarize_conversation(self, *args, **kwargs):
            return ""
        def update_memory(self, *args, **kwargs):
            return None
    
    ai_query_handler = DummyAIHandler()
    logger.warning("Using dummy AI handler - AI features will be limited")

# FastAPI

from fastapi import (
    APIRouter, 
    Depends, 
    HTTPException, 
    UploadFile as FastAPIFile, 
    File, 
    Request, 
    Form, 
    status,
    Path,
    Body,
    Header
)
from fastapi import UploadFile
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse, FileResponse, Response, StreamingResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, List, Optional, Any, Union

# Application imports
from . import services
from .utils import compute_sha256
from .db import User, File as DBFile, Query, Device, Session, SessionLocal
from .auth import (
    get_db, 
    get_password_hash, 
    authenticate_user, 
    create_access_token,
    get_current_user, 
    ACCESS_TOKEN_EXPIRE_MINUTES
)

# Import schemas
from .schemas.health import HealthCheckResponse
from .schemas import (
    RegisterRequest, 
    RegisterResponse, 
    LoginRequest, 
    TokenResponse,
    UserResponse,
    FileUploadResponse,
    QueryRequest,
    QueryResponse,
    DeviceHeartbeatRequest,
    DeviceInfo,
    DeviceListResponse,
    MessageRequest,
    MessageResponse,
    IncomingMessage,
    CallResponse,
    TranscriptionResponse,
    FileInfo,
    FileListResponse,
    FileUpdateRequest,
    FileType
)

# Initialize router
router = APIRouter(
    prefix="",
    tags=["api"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "Missing or invalid authentication"},
        status.HTTP_403_FORBIDDEN: {"description": "Not enough permissions"},
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
    },
)

# Initialize assets folder
ASSETS_FOLDER = "assets"
os.makedirs(ASSETS_FOLDER, exist_ok=True)

# Start the scheduler when the module loads
services.start_scheduler()

# ===========================================
# API Endpoints
# ===========================================



@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Check if the API is running and healthy.",
    tags=["system"]
)
async def health_check() -> HealthCheckResponse:
    """Check the health status of the API.
    
    This endpoint performs a basic health check of the API and its dependencies.
    
    Returns:
        HealthCheckResponse: The health check response containing status, timestamp, and version.
    """
    return HealthCheckResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="1.0.0"
    )


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account with the provided credentials.",
    tags=["auth"],
    responses={
        400: {"description": "Username or phone number already exists"},
        422: {"description": "Validation error in request data"}
    }
)
async def register(
    req: RegisterRequest, 
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Register a new user in the system.
    
    Creates a new user account with the provided username, password, and optional phone number.
    The password will be hashed before storage.
    
    Args:
        req: The registration request containing user details.
        db: Database session dependency.
        
    Returns:
        Dict[str, Any]: Registration confirmation with user ID and username.
        
    Raises:
        HTTPException: 400 if username or phone number already exists.
    """
    # Check if username already exists
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if phone number already exists (if provided)
    if req.phone_number and db.query(User).filter(User.phone_number == req.phone_number).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Phone number already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(req.password)
    db_user = User(
        username=req.username,
        hashed_password=hashed_password,
        phone_number=req.phone_number,
        is_active=True
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return {
        "user_id": db_user.id,
        "username": db_user.username,
        "message": "User registered successfully"
    }


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="User Login",
    description="Authenticate a user and retrieve an access token.",
    tags=["auth"],
    responses={
        400: {"description": "Incorrect username or password"},
        422: {"description": "Validation error in request data"}
    }
)
async def login(
    request: Request, 
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Authenticate a user, register the calling device and bootstrap a
    `<device_id>.json` tracking file that will be kept in-sync by the
    `/device/heartbeat` endpoint.

    The request **MUST** be sent as a regular OAuth2 password flow form but can
    optionally include the following extra form fields so we can properly
    identify the client device:
    - `device_name`: Human readable name (e.g. "John's MacBook")
    - `device_type`: One of `mac_app`, `chrome_extension`, *etc.*
    """
    # Parse the oauth2 form manually so that we can grab the extra fields
    form = await request.form()
    username: str = form.get("username")
    password: str = form.get("password")
    device_name: str = form.get("device_name") or None
    device_type: str = form.get("device_type") or None
    device_uuid: str = form.get("device_id") or None  # client-provided stable id

    # Guard: we prefer explicit identification; if missing, generate placeholders
    if not device_uuid:
        # Fall back to generated uuid to avoid duplicates of "unknown_device"
        device_uuid = str(uuid.uuid4())
    if not device_name:
        device_name = f"device_{device_uuid[:6]}"
    if not device_type:
        device_type = "unknown_type"

    user = authenticate_user(db, username, password)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")

    # ------------------------------------------------------------------
    # Register / update the device that is requesting the token
    # ------------------------------------------------------------------
    device = None
    if device_uuid:
        device = db.query(Device).filter(
            Device.userId == user.userId,
            Device.device_uuid == device_uuid
        ).first()
    if not device:
        device = db.query(Device).filter(
            Device.userId == user.userId,
            Device.device_name == device_name
        ).first()

    if not device:
        device = Device(
            userId=user.userId,
            device_name=device_name,
            device_type=device_type,
            device_uuid=device_uuid,
        )
        db.add(device)
        db.commit()
        db.refresh(device)
    else:
        # Keep server-side name/type up-to-date in case they changed on client
        device.device_name = device_name
        device.device_type = device_type
        if device_uuid and not device.device_uuid:
            device.device_uuid = device_uuid
        db.commit()

    # ------------------------------------------------------------------
    # Create an initial per-device tracking file (<device_id>.json)
    # ------------------------------------------------------------------
    filename = f"{device.deviceId}.json"
    file_entry = db.query(DBFile).filter(
        DBFile.userId == user.userId,
        DBFile.filename == filename
    ).first()

    if file_entry and file_entry.content:
        try:
            data = json.loads(file_entry.content.decode("utf-8"))
        except Exception:
            data = {"events": []}
    else:
        data = {
            "deviceId": device.deviceId,
            "device_name": device_name,
            "device_type": device_type,
            "events": []
        }

    # Append the current foreground information as a new event
    data.setdefault("events", []).append({
        "timestamp": datetime.utcnow().isoformat(),
        "current_app": None,
        "current_page": None,
        "current_url": None
    })

    updated_bytes = json.dumps(data).encode("utf-8")

    if file_entry:
        file_entry.content = updated_bytes
        file_entry.size = len(updated_bytes)
        file_entry.file_hash = compute_sha256(updated_bytes)
        file_entry.uploaded_at = datetime.utcnow()
    else:
        new_file = DBFile(
            filename=filename,
            userId=user.userId,
            size=len(updated_bytes),
            content=updated_bytes,
            file_hash=compute_sha256(updated_bytes),
            content_type="application/json",
            uploaded_at=datetime.utcnow()
        )
        db.add(new_file)

    db.commit()

    # ------------------------------------------------------------------
    # Generate the access token and respond
    # ------------------------------------------------------------------
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "deviceId": device.deviceId
    }

@router.post(
    "/upload",
    response_model=FileUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a file",
    description="Upload a file to the server. The file will be associated with the authenticated user's account.",
    tags=["files"],
    responses={
        400: {"description": "File too large or invalid"},
        401: {"description": "Not authenticated"},
        500: {"description": "Upload failed"}
    }
)
async def upload_file(
    file: UploadFile = File(..., description="The file to upload"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    try:
        contents = await file.read()
        file_size = len(contents)
        file_hash = compute_sha256(contents)

        if file_size > user.max_file_size:
            raise HTTPException(status_code=400, detail="File too large")

        existing = db.query(DBFile).filter(
            DBFile.userId == user.userId,
            DBFile.filename == file.filename
        ).first()

        content_type = mimetypes.guess_type(file.filename)[0] or "application/octet-stream"

        if existing:
            if existing.file_hash == file_hash:
                return {"message": "File already exists", "fileId": existing.fileId}
            else:
                existing.content = contents
                existing.size = file_size
                existing.file_hash = file_hash
                existing.uploaded_at = datetime.utcnow()
                existing.content_type = content_type
                db.commit()
                return {"message": "File updated", "fileId": existing.fileId}

        # New file
        new_file = DBFile(
            filename=file.filename,
            userId=user.userId,
            size=file_size,
            content=contents,
            file_hash=file_hash,
            content_type=content_type,
            uploaded_at=datetime.utcnow()
        )

        db.add(new_file)
        db.commit()
        db.refresh(new_file)

        return {
            "message": "File uploaded",
            "fileId": new_file.fileId,
            "size": new_file.size,
            "hash": new_file.file_hash
        }

    except Exception as e:
        db.rollback()
        log_error(f"Upload failed: {e}", e, endpoint="/upload")
        raise HTTPException(status_code=500, detail="Upload failed")

@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Process AI Query",
    description="""
    Process a user query using the AI model and return a response.
    Maintains conversation context using chat_id for multi-turn conversations.
    """,
    tags=["AI"],
    responses={
        400: {"description": "Invalid request parameters"},
        401: {"description": "Not authenticated"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "AI service error"}
    }
)
async def query_endpoint(
    query_data: QueryRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Process an AI query from a user.
    
    Sends the query to the OpenAI API and returns the response. The query and response
    are associated with a chat ID for maintaining conversation context.
    All queries and responses are stored in the database for future reference.
    
    Args:
        query_data: The query parameters including the user's message
        request: The HTTP request object
        user: The authenticated user making the query
        db: Database session dependency
        
    Returns:
        Dict containing the AI's response and metadata
        
    Raises:
        HTTPException: If there's an error processing the query
    """
    # First check if there's any content in the request body
    body_bytes = await request.body()
    if not body_bytes:
        return JSONResponse({"error": "Empty request body"}, status_code=400)
        
    # Try to parse the JSON body
    try:
        body = await request.json()
    except json.JSONDecodeError as json_err:
        return JSONResponse({"error": f"Invalid JSON in request body: {str(json_err)}"}, status_code=400)
        

    try:
        
        # Check for required fields
        if not body.get("query"):
            return JSONResponse(status_code=400, content={"error": "No query provided"})
            
        # Check for chat_id field - Make sure this runs BEFORE the try-except for the AI agent
        # Handle both camelCase (chatId) and snake_case (chat_id) formats for compatibility
        chat_id = body.get("chat_id") or body.get("chatId")
        if chat_id is None or chat_id == "":
            return JSONResponse(status_code=400, content={"error": "No chat ID provided"})
        
        user_query = body.get("query")
        # check pageContent field
        page_content = body.get("pageContent")
        if page_content :
            user_query += "\n\n" + "Here is the page content: " + page_content

        
        # Create a new query record in the database (without response yet)
        db_query = Query(
            userId=user.userId,
            chatId=chat_id,
            query_text=user_query
        )
        db.add(db_query)
        db.commit()
        db.refresh(db_query)
        
        # Load long-term memory from DB
        longterm_memory_file = db.query(DBFile).filter(DBFile.userId == user.userId, DBFile.filename == "long_term_memory.json").first()
        longterm_content_str = longterm_memory_file.content.decode('utf-8') if longterm_memory_file and longterm_memory_file.content else "{}"

        # Load short-term memory from DB
        shortterm_memory_file = db.query(DBFile).filter(DBFile.userId == user.userId, DBFile.filename == "short_term_memory.json").first()
        shortterm_content_str = shortterm_memory_file.content.decode('utf-8') if shortterm_memory_file and shortterm_memory_file.content else "{}"
        
        # Initialize memory managers with parsed JSON content
        try:
            longterm_memory_data = json.loads(longterm_content_str)
            shortterm_memory_data = json.loads(shortterm_content_str)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding memory content from DB for user {user.userId}: {e}. LTM content: '{longterm_content_str[:200]}', STM content: '{shortterm_content_str[:200]}'")
            # Fallback to empty memory if JSON is corrupted
            longterm_memory_data = {}
            shortterm_memory_data = {}

        long_term_memory = LongTermMemoryManager(memory_content=longterm_memory_data)
        short_term_memory = ShortTermMemoryManager(memory_content=shortterm_memory_data)

        # Include user markdown notes in the query context
        try:
            notes_files = (
                db.query(DBFile)
                .filter(DBFile.userId == user.userId, DBFile.filename.ilike('%.md'))
                .limit(50)
                .all()
            )
            notes_parts = []
            for nf in notes_files:
                try:
                    note_text = (nf.content or b'').decode('utf-8', errors='ignore')
                except Exception:
                    note_text = ''
                notes_parts.append(f"{nf.filename}_{note_text}")
            if notes_parts:
                concatenated_notes = "_".join(notes_parts)
                user_query += "\n\nthese are the notes of the user:" + concatenated_notes
        except Exception as e:
            logger.error(f"Error loading user notes for query: {e}")

        # Call your AI agent with try/except to handle Vercel environment limitations
        try:
            from aiagent.handler.query import query_openai
            
            # Set up auxiliary data for the AI query
            aux_data = {
                "username": user.username,
                "user_id": user.userId,
                "chat_id": chat_id,
                "query_id": db_query.queryId
            }
            
            # Read references
            from aiagent.context.reference import read_references
            try:
                references = read_references()
            except Exception as e:
                logging.error(f"Error loading references: {e}")
                references = {}
            
            # Send query to AI agent using query_openai instead of ask_ai
            response = query_openai(
                query=user_query,
                long_term_memory=long_term_memory,
                short_term_memory=short_term_memory,
                aux_data=aux_data,
                references=references,
                max_tokens=1000,  # Default max tokens
                temperature=0.7  # Default temperature
            )
            
            # If the query was successful, update the memory
            if not response.startswith("Error:"):
                # Update shortterm memory
                conversations = shortterm_memory_data.get("conversations", [])
                # Create a summary
                from aiagent.handler.query import summarize_conversation, update_memory
                summary = summarize_conversation(user_query, response)
                updated = update_memory(user_query, response, long_term_memory) 
                
                # Update conversations
                shortterm_memory_data["conversations"] = conversations + [{
                    "query": user_query, 
                    "response": response, 
                    "summary": summary
                }]

                # save updated longterm memory
                if updated : 
                    log_something("Updated longterm memory:"+str(long_term_memory.get_content()), "queryEndpoint")
                    if longterm_memory_file:
                        updated_content = long_term_memory.get_content()
                        longterm_memory_file.content = json.dumps(updated_content).encode('utf-8')
                        longterm_memory_file.size = len(longterm_memory_file.content)
                        # db.add(longterm_memory_file)
                        db.commit()
                    else:

                        # long_term_memory.json didn't exist for this user, create it now
                        new_longterm_file = DBFile(
                            filename="long_term_memory.json",
                            userId=user.userId,
                            content=json.dumps(long_term_memory.get_content()).encode('utf-8'),
                            content_type="application/json"
                        )
                        new_longterm_file.size = len(new_longterm_file.content)
                        db.add(new_longterm_file)
                        db.commit()
                else :
                    log_something("Longterm memory not updated", "queryEndpoint")
                
                # Save updated shortterm memory back to the database
                if shortterm_memory_file:
                    shortterm_memory_file.content = json.dumps(shortterm_memory_data).encode('utf-8')
                    shortterm_memory_file.size = len(shortterm_memory_file.content)
                else:
                    # short_term_memory.json didn't exist for this user, create it now
                    new_shortterm_file = DBFile(
                        filename="short_term_memory.json",
                        userId=user.userId,
                        content=json.dumps(shortterm_memory_data).encode('utf-8'),
                        content_type="application/json"
                    )
                    new_shortterm_file.size = len(new_shortterm_file.content)
                    db.add(new_shortterm_file)
                db.commit()

        except FileNotFoundError as file_err:
            # Handle missing files in Vercel environment
            log_error(f"AI agent file not found: {str(file_err)}", exc=file_err, endpoint="/query")
            if os.environ.get("VERCEL"):
                # In Vercel, return a graceful error message for testing purposes
                response = "The AI agent is not fully configured in this environment. This is a test instance."
            else:
                # In non-Vercel environments, still raise the error
                raise file_err
        
        # Log the response
        log_ai_response(response, "/query")
        
        # Update the query record with the response
        db_query.response = response
        db.commit()
        
        return {
            "response": response,
            "query": user_query,
            "chat_id": chat_id,
            "queryId": db_query.queryId
        }
    except Exception as e:
        db.rollback()  # Rollback transaction on error
        log_error(f"AI query failed: {str(e)}", exc=e, endpoint="/query")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/active")
async def update_active_item(
    request: Request, 
    user: User = Depends(get_current_user),
    db: SessionLocal = Depends(get_db)
):
    """Update or add an active item for the authenticated user.
    
    Records the device, path, and title of the active item being viewed.
    If an entry with the same device exists, it will be updated.
    
    Args:
        request: The HTTP request containing the active item data
        user: The authenticated user
        db: Database session
        
    Returns:
        dict: Success status and updated active items
        
    Raises:
        HTTPException: 400 if required fields are missing
        HTTPException: 404 if user's short-term memory is not found
    """
    try:
        # Log request start
        log_request_start('/active', request.method, dict(request.headers), request.client.host if request.client else None)
        
        # Parse request body
        try:
            data = await request.json()
            log_request_payload(data, '/active')
        except json.JSONDecodeError as json_err:
            log_error(f"Invalid JSON: {str(json_err)}", json_err, {"endpoint": "/active"}, "/active")
            return JSONResponse({"error": f"Invalid JSON in request body: {str(json_err)}"}, status_code=400)
            
        # Validate required fields
        device = data.get('device')
        path = data.get('path', '')
        title = data.get('title', '')
        
        if not device:
            log_error("Missing device identifier", None, {"endpoint": "/active"}, "/active")
            return JSONResponse({"error": "Missing device identifier"}, status_code=400)
            
        # Get user's short-term memory file
        stm_file = db.query(DBFile).filter(
            DBFile.userId == user.userId,
            DBFile.filename == "short_term_memory.json"
        ).first()
        
        if not stm_file:
            log_error("Short-term memory not found", None, {"userId": user.userId}, "/active")
            raise HTTPException(status_code=404, detail="Short-term memory not found")
            
        # Parse existing memory
        try:
            memory = json.loads(stm_file.content.decode('utf-8'))
        except json.JSONDecodeError as e:
            log_error(f"Invalid short-term memory format: {str(e)}", e, {"userId": user.userId}, "/active")
            raise HTTPException(status_code=500, detail="Invalid short-term memory format")
            
        # Initialize active list if it doesn't exist
        if 'active' not in memory:
            memory['active'] = []
            
        # Create new active item
        new_item = {
            'device': device,
            'path': path,
            'title': title,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Update existing item or add new one
        updated = False
        for i, item in enumerate(memory['active']):
            if item.get('device') == device:
                memory['active'][i] = new_item
                updated = True
                break
                
        if not updated:
            memory['active'].append(new_item)
            
        # Update the file in database
        stm_file.content = json.dumps(memory).encode('utf-8')
        stm_file.size = len(stm_file.content)
        db.commit()
        
        # Log success
        #logger.info(f"[SERVER] Updated active item for user {user.userId}, device {device}")
        
        # Return success response with updated active items
        response = {
            "status": "success",
            "active": memory['active']
        }
        log_response(200, response, '/active')
        return JSONResponse(response, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(str(e), e, {"endpoint": "/active"}, "/active")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/files")
def list_files(user: User = Depends(get_current_user), db: SessionLocal = Depends(get_db)):
    """List all files uploaded by the authenticated user.
    
    Retrieves files from the database instead of checking the filesystem directly.
    
    Args:
        user: The authenticated user whose files to list
        db: Database session dependency
        
    Returns:
        dict: List of files with their metadata
    """
    try:
        # Query the database for files owned by this user
        files = db.query(DBFile).filter(DBFile.userId == user.userId).all()
        
        # Format the response with file metadata
        file_list = [
            {
                "fileId": file.fileId,
                "filename": file.filename,
                "size": file.size,
                "uploaded_at": file.uploaded_at
            } for file in files
        ]
        
        return {"files": file_list, "count": len(file_list)}
    except Exception as e:
        log_error(f"Error listing files: {str(e)}", exc=e, endpoint="/files")
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

@router.get("/download/{fileId}")
def download_file(fileId: int, user: User = Depends(get_current_user), db: SessionLocal = Depends(get_db)):
    """Download a specific file by its ID.
    
    Retrieves the file record from the database before accessing the filesystem.
    
    Args:
        fileId: The ID of the file to download
        user: The authenticated user requesting the download
        db: Database session dependency
        
    Returns:
        FileResponse: The file content as a download
        
    Raises:
        HTTPException: 404 if file not found
        HTTPException: 403 if trying to access another user's file
    """
    try:
        # Query the database for the file record
        file_record = db.query(DBFile).filter(DBFile.fileId == fileId).first()
        
        # Check if file exists and belongs to the user
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")
            
        if file_record.userId != user.userId:
            raise HTTPException(status_code=403, detail="You don't have permission to access this file")
        
        # First try to get content directly from the database
        if file_record.content is not None:
            file_content = file_record.content
            content_type = file_record.content_type or "application/octet-stream"
        else:
            # Fall back to the filesystem if content is not in the database
            filepath = file_record.path
            if filepath and os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    file_content = f.read()
                content_type = file_record.content_type or "application/octet-stream"
            else:
                # Try standard path as fallback
                user_folder = os.path.join(ASSETS_FOLDER, str(user.userId))
                fallback_path = os.path.join(user_folder, file_record.filename)
                if os.path.exists(fallback_path):
                    with open(fallback_path, "rb") as f:
                        file_content = f.read()
                    content_type = file_record.content_type or "application/octet-stream"
                else:
                    raise HTTPException(status_code=404, detail="File content not found")
        
        # Guess content type from filename if not set
        if not content_type or content_type == "application/octet-stream":
            import mimetypes
            content_type = mimetypes.guess_type(file_record.filename)[0] or "application/octet-stream"
        
        # Stream the file from memory
        def iterfile():
            yield file_content
                
        return StreamingResponse(
            iterfile(), 
            media_type=content_type, 
            headers={"Content-Disposition": f"attachment; filename={quote(file_record.filename)}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error downloading file: {str(e)}", exc=e, endpoint=f"/download/{fileId}")
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

@router.delete("/delete/{fileId}")
def delete_file(fileId: int, user: User = Depends(get_current_user), db: SessionLocal = Depends(get_db)):
    """Delete a specific file by its ID.
    
    Deletes both the database record and the file on disk.
    
    Args:
        fileId: The ID of the file to delete
        user: The authenticated user requesting the deletion
        db: Database session dependency
        
    Returns:
        dict: Confirmation message
        
    Raises:
        HTTPException: 404 if file not found
        HTTPException: 403 if trying to delete another user's file
    """
    try:
        # Query the database for the file record
        file_record = db.query(DBFile).filter(DBFile.fileId == fileId).first()
        
        # Check if file exists and belongs to the user
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")
            
        if file_record.userId != user.userId:
            raise HTTPException(status_code=403, detail="You don't have permission to delete this file")
        
        # Get file information before deleting
        filename = file_record.filename
        filepath = file_record.path
        
        # Delete the file from disk if it exists
        try:
            # If filepath is None, we're using database storage, so no need to remove from disk
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
            else:
                # Try the standard path pattern as fallback (only for non-Vercel environments)
                if not os.environ.get("VERCEL") and not os.environ.get("READ_ONLY_FS"):
                    user_folder = os.path.join(ASSETS_FOLDER, str(user.userId))
                    fallback_path = os.path.join(user_folder, filename)
                    if os.path.exists(fallback_path):
                        os.remove(fallback_path)
        except (OSError, TypeError) as e:
            # Continue even if file removal fails, as we still want to remove the database record
            log_error(f"Error removing file from disk: {str(e)}", exc=e, endpoint=f"/delete/{fileId}")
        
        # Delete the database record
        db.delete(file_record)
        db.commit()
        
        return {"message": f"File '{filename}' deleted successfully.", "fileId": fileId}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log_error(f"Error deleting file: {str(e)}", exc=e, endpoint=f"/delete/{fileId}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


@router.get('/profile', response_model=UserResponse)
def profile(current_user: User = Depends(get_current_user)):
    """Get profile information for the currently authenticated user.
    
    Returns:
        UserResponse: The authenticated user's profile data, including their role and phone number.
    """
    # The current_user object (SQLAlchemy model) will be automatically
    # converted to a UserResponse Pydantic model by FastAPI.
    return UserResponse(
        userId=current_user.userId,
        username=current_user.username,
        max_file_size=current_user.max_file_size,
        role=current_user.role,
        phone_number=current_user.phone_number
    )



# --- Device Endpoints ---


@router.post("/device/heartbeat")
async def device_heartbeat(
    device_id: str = Form(None),  # client stable UUID (device_uuid)
    device_name: str = Form(None),
    device_type: str = Form(None),
    current_app: str = Form(None),
    current_page: str = Form(None),
    current_url: str = Form(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    now = datetime.utcnow()

    # Resolve device record prioritising device_uuid when supplied
    device = None
    if device_id:
        device = db.query(Device).filter(
            Device.userId == user.userId,
            Device.device_uuid == device_id
        ).first()

    if not device and device_name:
        device = db.query(Device).filter(
            Device.userId == user.userId,
            Device.device_name == device_name
        ).first()

    # Create the device entry on first heartbeat if it was not registered
    if not device:
        # Fallback defaults
        if not device_name:
            device_name = f"device_{device_id[:6] if device_id else uuid.uuid4().hex[:6]}"
        if not device_type:
            device_type = "unknown_type"

        device = Device(
            userId=user.userId,
            device_name=device_name,
            device_type=device_type,
            device_uuid=device_id,
        )
        db.add(device)
        db.commit()
        db.refresh(device)

    # Update the last-seen timestamp
    device.last_seen = now
    db.commit()

    # ------------------------------------------------------------------
    # Persist foreground context to the <device_id>.json file
    # ------------------------------------------------------------------
    filename = f"{device.deviceId}.json"
    file_entry = db.query(DBFile).filter(
        DBFile.userId == user.userId,
        DBFile.filename == filename
    ).first()

    if file_entry and file_entry.content:
        try:
            data = json.loads(file_entry.content.decode("utf-8"))
        except Exception:
            data = {"events": []}
    else:
        data = {
            "deviceId": device.deviceId,
            "device_name": device_name,
            "device_type": device_type,
            "events": []
        }

    # Append the current foreground information as a new event
    data.setdefault("events", []).append({
        "timestamp": now.isoformat(),
        "current_app": current_app,
        "current_page": current_page,
        "current_url": current_url
    })

    updated_bytes = json.dumps(data).encode("utf-8")

    if file_entry:
        file_entry.content = updated_bytes
        file_entry.size = len(updated_bytes)
        file_entry.file_hash = compute_sha256(updated_bytes)
        file_entry.uploaded_at = now
    else:
        new_file = DBFile(
            filename=filename,
            userId=user.userId,
            size=len(updated_bytes),
            content=updated_bytes,
            file_hash=compute_sha256(updated_bytes),
            content_type="application/json",
            uploaded_at=now
        )
        db.add(new_file)

    db.commit()

    return {
        "message": "Device heartbeat received and state updated",
        "deviceId": device.deviceId,
        "device_uuid": device.device_uuid,
        "last_seen": device.last_seen.isoformat()
    }


@router.post("/device/logout")
async def device_logout(
    device_id: str = Form(None),
    device_name: str = Form(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Resolve device record prioritising device_uuid when supplied
    device = None
    if device_id:
        device = db.query(Device).filter(
            Device.userId == user.userId,
            Device.device_uuid == device_id
        ).first()

    if not device and device_name:
        device = db.query(Device).filter(
            Device.userId == user.userId,
            Device.device_name == device_name
        ).first()

    if device:
        # Mark device as offline by moving last_seen outside the online threshold
        offline_time = datetime.utcnow() - timedelta(minutes=10)
        device.last_seen = offline_time
        logger.info(f"[device_logout] Marked device '{device.device_name}' as offline at {offline_time.isoformat()}")
        db.commit()

    return {
        "status": "logged out",
        "deviceId": device.deviceId if device else None,
        "device_uuid": device.device_uuid if device else None,
    }


@router.get("/devices")
async def list_devices(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List all devices registered for the authenticated user.

    A device is considered **online** if its `last_seen` timestamp is within
    the last five minutes. Otherwise, it's treated as offline.
    """
    try:
        threshold = datetime.utcnow() - timedelta(minutes=5)
        devices = db.query(Device).filter(Device.userId == user.userId).all()

        device_list = []
        for device in devices:
            online = bool(device.last_seen and device.last_seen > threshold)
            device_list.append({
                "deviceId": device.deviceId,
                "name": device.device_name,
                "type": device.device_type,
                "last_seen": device.last_seen.isoformat() if device.last_seen else None,
                "online": online,
            })

        return {"devices": device_list, "count": len(device_list)}

    except Exception as e:
        log_error(f"Error listing devices: {str(e)}", exc=e, endpoint="/devices")
        raise HTTPException(status_code=500, detail="Error listing devices")


# --- Twilio Webhook Endpoints ---

@router.post("/api/webhooks/twilio/incoming-message")
async def handle_twilio_incoming_message(
    request: Request, 
    db: Session = Depends(get_db)
):
    """
    Handles incoming SMS messages from Twilio, processes them via AI, and sends a reply.
    Twilio sends data as 'application/x-www-form-urlencoded'.
    """
    client_host = request.client.host if request.client else "unknown_client"
    endpoint_name = "/api/webhooks/twilio/incoming-message"
    
    # Initialize variables
    user_query_text = ""
    found_user = None
    normalized_from_number_str = ""
    From = ""
    
    try:
        # Log the incoming request
        logger.info(f"[{endpoint_name}] Received request from {client_host}")
        
        # Get form data
        form_data = await request.form()
        logger.info(f"[{endpoint_name}] Raw form data: {dict(form_data)}")
        
        # Extract required fields
        From = form_data.get('From', '')
        Body = form_data.get('Body', '')
        logger.info(f"[{endpoint_name}] From: {From}, Body: {Body}")
        
        if not From:
            logger.warning(f"[{endpoint_name}] Missing 'From' parameter in request")
            return Response(
                content="<Response><Message>Error: Missing sender information</Message></Response>",
                media_type="application/xml",
                status_code=400
            )
            
        # Normalize phone number
        normalized_from_number_str = re.sub(r'\D', '', From)
        if not normalized_from_number_str:
            logger.warning(f"[{endpoint_name}] Could not extract phone number from: {From}")
            return Response(
                content="<Response><Message>Sorry, we could not identify your phone number.</Message></Response>",
                media_type="application/xml",
                status_code=200
            )

        # Look up user by phone number
        logger.info(f"[{endpoint_name}] Looking up user with phone number: {normalized_from_number_str}")
        try:
            phone_number_to_lookup = int(normalized_from_number_str)
            found_user = db.query(User).filter(User.phone_number == phone_number_to_lookup).first()
        except ValueError as ve:
            logger.error(f"[{endpoint_name}] Invalid phone number format: {normalized_from_number_str}", exc_info=True)
            return Response(
                content="<Response><Message>Invalid phone number format.</Message></Response>",
                media_type="application/xml",
                status_code=200
            )
        
        if not found_user:
            logger.info(f"[{endpoint_name}] No user found for number: {normalized_from_number_str}")
            return Response(
                content="<Response><Message>Sorry, we couldn't find an account with this phone number.</Message></Response>",
                media_type="application/xml",
                status_code=200
            )
            
        logger.info(f"[{endpoint_name}] Found user: {found_user.username} (ID: {found_user.userId})")
        
        # Extract the message body which contains the user's query
        user_query_text = Body.strip() if Body else ""
        
        if not user_query_text:
            logger.warning(f"[{endpoint_name}] Empty message body from user {found_user.userId}")
            return Response(
                content="<Response><Message>Please provide a message with your query.</Message></Response>",
                media_type="application/xml",
                status_code=200
            )
            
    except Exception as e:
        logger.error(f"[{endpoint_name}] Unexpected error during request processing: {str(e)}", exc_info=True)
        return Response(
            content="<Response><Message>An unexpected error occurred. Please try again later.</Message></Response>",
            media_type="application/xml",
            status_code=500
        )
    
    # If we get here, we have a valid user and query text
    try:
        # Create a new query record in the database
        db_query = Query(
            userId=user.userId,
            chatId=chat_id,
            query_text=user_query_text
        )
        db.add(db_query)
        db.commit()
        db.refresh(db_query)
        log_ai_call(user_query_text, "default_sms_model", endpoint_name)

        # Retrieve memory files
        shortterm_file_db = db.query(DBFile).filter(
            DBFile.userId == user.userId, DBFile.filename == "short_term_memory.json"
        ).first()
        longterm_file_db = db.query(DBFile).filter(
            DBFile.userId == user.userId, DBFile.filename == "long_term_memory.json"
        ).first()

        shortterm_content = json.loads(shortterm_file_db.content.decode('utf-8')) if shortterm_file_db and shortterm_file_db.content else {}
        longterm_content = json.loads(longterm_file_db.content.decode('utf-8')) if longterm_file_db and longterm_file_db.content else {}

        short_term_memory = ShortTermMemoryManager(memory_content=shortterm_content)
        long_term_memory = LongTermMemoryManager(memory_content=longterm_content)

        aux_data = {
            "username": user.username,
            "user_id": user.userId,
            "chat_id": chat_id,
            "query_id": db_query.queryId,
            "client_info": {
                "max_tokens": 1024,
                "temperature": 0.7
            }
        }
        
        # references = read_references()

        ai_response = ai_query_handler.query_openai(
            query=user_query_text,
            long_term_memory=long_term_memory,
            short_term_memory=short_term_memory,
            aux_data=aux_data,
            max_tokens=aux_data["client_info"]["max_tokens"],
            temperature=aux_data["client_info"]["temperature"],
            # references=references,
        )
        log_ai_response(ai_response, endpoint_name)

        db_query.response = ai_response # Store AI response
        db.commit()

        if not ai_response.startswith("Error:"):
            summary = ai_query_handler.summarize_conversation(user_query_text, ai_response)
            updated_ltm = ai_query_handler.update_memory(user_query_text, ai_response, long_term_memory)

            current_conversations = shortterm_content.get("conversations", [])
            shortterm_content["conversations"] = current_conversations + [{
                "query": user_query_text, "response": ai_response, "summary": summary
            }]

            # limit conversations to 20
            if len(shortterm_content["conversations"]) > 20:
                shortterm_content["conversations"] = shortterm_content["conversations"][-20:]
            
            
            if shortterm_file_db:
                shortterm_file_db.content = json.dumps(shortterm_content).encode('utf-8')
                shortterm_file_db.size = len(shortterm_file_db.content)
            else: # Should not happen if user registration creates it
                #logger.warning(f"[{endpoint_name}] short_term_memory.json not found for user {user.userId}, creating new.")
                # Create if missing logic might be needed here
                pass 

            if updated_ltm and longterm_file_db:
                longterm_file_db.content = json.dumps(long_term_memory.get_content()).encode('utf-8')
                longterm_file_db.size = len(longterm_file_db.content)
                db.add(longterm_file_db)
            db.commit()
        
        twiml_reply = f"<Response><Message>{ai_response}</Message></Response>"
        log_response(200, "TwiML reply sent", endpoint_name)
        return Response(content=twiml_reply, media_type="application/xml", status_code=200)

    except Exception as e:
        #logger.error(f"[{endpoint_name}] Error processing AI query for SMS: {e}", exc_info=True)
        db.rollback() # Rollback any partial DB changes on error
        twiml_error_reply = "<Response><Message>Sorry, an internal error occurred while processing your message.</Message></Response>"
        return Response(content=twiml_error_reply, media_type="application/xml", status_code=500)


@router.post("/api/webhooks/twilio/message-status")
async def handle_twilio_message_status(
    request: Request, 
    MessageSid: str = Form(...), 
    MessageStatus: str = Form(...),
    To: str = Form(None),
    From: str = Form(None),
    ErrorCode: str = Form(None)
):
    """
    Handles delivery status updates for outbound messages from Twilio.
    
    Args:
        MessageSid: The unique ID of the message
        MessageStatus: The delivery status (queued, failed, sent, delivered, etc.)
        To: The recipient's phone number
        From: The sender's phone number
        ErrorCode: Error code if message failed to deliver
    """
    endpoint_name = "/api/webhooks/twilio/message-status"
    client_host = request.client.host if request.client else "unknown_client"
    
    try:
        log_request_start(endpoint_name, "POST", dict(request.headers), client_host)
        
        # Log the message status update
        log_info(
            f"Twilio message status update - SID: {MessageSid}, "
            f"Status: {MessageStatus}, To: {To}, From: {From}, "
            f"Error: {ErrorCode if ErrorCode else 'None'}",
            endpoint=endpoint_name
        )
        
        # Here you could update your database with the message status
        # For example:
        # update_message_status_in_db(MessageSid, MessageStatus, error_code=ErrorCode)
        
        # Log successful processing
        log_response(200, "Message status updated", endpoint_name)
        return Response(status_code=200)
        
    except Exception as e:
        log_error(
            f"Error processing message status update - SID: {MessageSid}, Error: {str(e)}",
            exc=e,
            endpoint=endpoint_name
        )
        # Even if there's an error processing the status update,
        # we should still return 200 to acknowledge receipt to Twilio
        return Response(status_code=200)



@router.post("/api/webhooks/twilio/incoming-call")
async def handle_twilio_incoming_call(
    request: Request,
    From: str = Form(...),
    db: Session = Depends(get_db)
):
    normalized_from = re.sub(r"\D", "", From)
    user = db.query(User).filter(User.phone_number == int(normalized_from)).first() if normalized_from.isdigit() else None

    if not user:
        twiml = """
        <Response>
            <Say voice="alice">Sorry, you are not recognized by Gad.</Say>
            <Hangup/>
        </Response>
        """
        return Response(content=twiml.strip(), media_type="application/xml", status_code=200)

    username = user.username

    twiml = f"""
    <Response>
        <Say voice="alice">Hi {username}, how can I help you today?</Say>
        <Record 
            action="/api/webhooks/twilio/incoming-call" 
            transcribe="true"
            transcribeCallback="/api/webhooks/twilio/transcription-callback"
            maxLength="30"
            timeout="5"
            playBeep="true" />
    </Response>
    """
    return Response(content=twiml.strip(), media_type="application/xml", status_code=200)


@router.post("/api/webhooks/twilio/transcription-callback")
async def handle_transcription_callback(
    request: Request,
    From: str = Form(...),
    TranscriptionText: str = Form(...),
    RecordingUrl: str = Form(None),
    db: Session = Depends(get_db)
):
    normalized_from = re.sub(r"\D", "", From)
    user = db.query(User).filter(User.phone_number == int(normalized_from)).first() if normalized_from.isdigit() else None

    if not user:
        return Response(status_code=204)

    chat_id = f"voice_{normalized_from}"
    user_transcript = TranscriptionText

    # Save query
    db_query = Query(userId=user.userId, chatId=chat_id, query_text=user_transcript)
    db.add(db_query)
    db.commit()
    db.refresh(db_query)

    # Load memory
    shortterm = db.query(DBFile).filter(DBFile.userId == user.userId, DBFile.filename == "short_term_memory.json").first()
    longterm = db.query(DBFile).filter(DBFile.userId == user.userId, DBFile.filename == "long_term_memory.json").first()
    short_content = json.loads(shortterm.content.decode('utf-8')) if shortterm and shortterm.content else {}
    long_content = json.loads(longterm.content.decode('utf-8')) if longterm and longterm.content else {}

    short_term_memory = ShortTermMemoryManager(memory_content=short_content)
    long_term_memory = LongTermMemoryManager(memory_content=long_content)

    aux_data = {
        "username": user.username,
        "user_id": user.userId,
        "chat_id": chat_id,
        "query_id": db_query.queryId,
        "client_info": {"max_tokens": 256, "temperature": 0.5}
    }

    ai_response = ai_query_handler.query_openai(
        query=user_transcript,
        long_term_memory=long_term_memory,
        short_term_memory=short_term_memory,
        aux_data=aux_data,
        max_tokens=256,
        temperature=0.5,
    )

    db_query.response = ai_response
    db.commit()

    summary = ai_query_handler.summarize_conversation(user_transcript, ai_response)
    short_content.setdefault("conversations", []).append({
        "query": user_transcript, "response": ai_response, "summary": summary
    })
    if shortterm:
        shortterm.content = json.dumps(short_content).encode('utf-8')
        shortterm.size = len(shortterm.content)
        db.commit()

    # Optionally send an SMS response or set up Twilio to redirect for the next round
    # Or ignore (Twilio will just hang up or timeout)
    return Response(status_code=200)



# send_twilio_message has been moved to services.py
