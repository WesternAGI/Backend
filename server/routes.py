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

from twilio.twiml.voice_response import VoiceResponse


# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)


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
    ACCESS_TOKEN_EXPIRE_MINUTES,
    verify_password
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
    "/token",
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
    db: Session = Depends(get_db),
    device_id: str = Form(None),
    device_name: str = Form(None),
    device_type: str = Form(None)
) -> Dict[str, Any]:
    logger.info(f"[LOGIN] Login attempt with device_id={device_id}, device_name={device_name}, device_type={device_type}")
    logger.info(f"[LOGIN] Request URL: {request.url}")
    logger.info(f"[LOGIN] Request headers: {dict(request.headers)}")
    
    try:
        # Parse the oauth2 form manually so that we can grab the extra fields
        form = await request.form()
        logger.info(f"[LOGIN] Form data received: {dict(form)}")
        
        username: str = form.get("username")
        password: str = form.get("password")
        device_name: str = form.get("device_name") or None
        device_type: str = form.get("device_type") or None
        device_uuid: str = form.get("device_id") or device_id  # client-provided stable id
        
        logger.info(f"[LOGIN] Parsed credentials - username: {username}, device_uuid: {device_uuid}")
        
        if not username or not password:
            error_msg = "Username and password are required"
            logger.error(f"[LOGIN] {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Missing credentials",
                    "message": error_msg,
                    "hint": "Please provide both username and password"
                }
            )
            
        logger.info(f"[LOGIN] Attempting to authenticate user: {username}")
        
        # Get user from database
        user = db.query(User).filter(User.username == username).first()
        if not user:
            error_msg = f"User not found: {username}"
            logger.warning(f"[LOGIN] {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Authentication failed",
                    "message": "Incorrect username or password",
                    "hint": "Please check your credentials and try again"
                }
            )
            
        logger.info(f"[LOGIN] Found user in database: {user.userId}")
        
        # Verify password
        if not verify_password(password, user.hashed_password):
            error_msg = f"Invalid password for user: {username}"
            logger.warning(f"[LOGIN] {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Authentication failed",
                    "message": "Incorrect username or password",
                    "hint": "Please check your credentials and try again"
                }
            )
            
        logger.info(f"[LOGIN] Password verified for user: {user.userId}")
        
        # Generate access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, 
            expires_delta=access_token_expires
        )
        
        logger.info(f"[LOGIN] Generated access token for user: {user.userId}")
        
        # Update or create device record
        if device_uuid:
            try:
                # Try to validate the device_uuid if it's provided
                uuid.UUID(device_uuid)
                
                device = db.query(Device).filter(
                    Device.userId == user.userId,
                    Device.device_uuid == device_uuid
                ).first()
                
                now = datetime.utcnow()
                
                if device:
                    # Update existing device
                    device.last_seen = now
                    device.online = True
                    if device_name:
                        device.device_name = device_name
                    if device_type:
                        device.device_type = device_type
                    logger.info(f"[LOGIN] Updated existing device: {device.deviceId} - {device.device_name}")
                else:
                    # Create new device
                    device = Device(
                        userId=user.userId,
                        device_uuid=device_uuid,
                        device_name=device_name or "Unknown Device",
                        device_type=device_type or "unknown",
                        last_seen=now,
                        online=True
                    )
                    db.add(device)
                    logger.info(f"[LOGIN] Created new device: {device_uuid} - {device_name or 'Unnamed Device'}")
                
                db.commit()
                logger.info(f"[LOGIN] Device record {'created' if not device.deviceId else 'updated'} successfully")
                
            except ValueError as e:
                logger.error(f"[LOGIN] Invalid device UUID format: {device_uuid} - {str(e)}")
                # Don't fail the login, just log the error
                device = None
        else:
            logger.warning("[LOGIN] No device_id provided in login request")
            device = None
        
        # Get user role (default to 'user' if not set)
        user_role = getattr(user, 'role', 'user')
        
        # Prepare response according to TokenResponse model
        response_data = {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": int(access_token_expires.total_seconds()),
            "user_id": user.userId,
            "username": user.username,
            "role": user_role,
            # These fields are not in the TokenResponse model but we'll keep them
            # for backward compatibility
            "device_id": device.device_uuid if device else None,
            "device_name": device.device_name if device else None
        }
        
        logger.info(f"[LOGIN] Login successful for user: {user.userId}")
        return response_data
        
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions as they're already properly formatted
        logger.error(f"[LOGIN] HTTP Exception: {str(http_exc)}")
        raise http_exc
        
    except Exception as e:
        # Log any unexpected errors
        error_msg = f"Unexpected error during login: {str(e)}"
        logger.error(f"[LOGIN] {error_msg}", exc_info=True)
        
        # Return a 500 error with detailed information in development
        import traceback
        error_detail = {
            "error": "Internal Server Error",
            "message": "An unexpected error occurred during login",
            "detail": str(e),
            "traceback": traceback.format_exc() if os.getenv("ENV") == "development" else None
        }
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail
        )
    
    logger.info(f"[LOGIN] Form data - username: {username}, device_uuid: {device_uuid}, device_name: {device_name}, device_type: {device_type}")

    # Guard: we prefer explicit identification; if missing, generate placeholders
    if not device_uuid:
        # Fall back to generated uuid to avoid duplicates of "unknown_device"
        device_uuid = str(uuid.uuid4())
        logger.info(f"[LOGIN] Generated new device_uuid: {device_uuid}")
    if not device_name:
        device_name = f"device_{device_uuid[:6]}"
        logger.info(f"[LOGIN] Using generated device_name: {device_name}")
    if not device_type:
        device_type = "unknown_type"
        logger.info(f"[LOGIN] Using default device_type: {device_type}")

    user = authenticate_user(db, username, password)
    if not user:
        logger.warning(f"[LOGIN] Authentication failed for username: {username}")
        raise HTTPException(status_code=400, detail="Invalid credentials")
        
    logger.info(f"[LOGIN] User authenticated: user_id={user.userId}, username={user.username}")

    # ------------------------------------------------------------------
    # Register / update the device that is requesting the token
    # ------------------------------------------------------------------
    logger.info(f"[LOGIN] Looking up device for user_id={user.userId}, device_uuid={device_uuid}")
    
    try:
        device = None
        
        # First try to find by device_uuid if provided
        if device_uuid:
            device = db.query(Device).filter(
                Device.userId == user.userId,
                Device.device_uuid == device_uuid
            ).first()
            if device:
                logger.info(f"[LOGIN] Found existing device by UUID: device_id={device.deviceId}, name={device.device_name}")
        
        # If not found by UUID, try by name
        if not device and device_name:
            logger.info(f"[LOGIN] Device not found by UUID, trying by name: {device_name}")
            device = db.query(Device).filter(
                Device.userId == user.userId,
                Device.device_name == device_name
            ).first()
            if device:
                logger.info(f"[LOGIN] Found existing device by name: device_id={device.deviceId}, uuid={device.device_uuid}")

        # If still not found, create a new device
        if not device:
            # Ensure device_uuid is unique
            if device_uuid:
                # Check if device_uuid already exists for another user
                existing = db.query(Device).filter(
                    Device.device_uuid == device_uuid,
                    Device.userId != user.userId
                ).first()
                if existing:
                    logger.warning(f"[LOGIN] Device UUID {device_uuid} already in use by another user")
                    device_uuid = str(uuid.uuid4())  # Generate a new UUID
                    logger.info(f"[LOGIN] Generated new UUID for device: {device_uuid}")
            
            device = Device(
                userId=user.userId,
                device_uuid=device_uuid,
                device_name=device_name,
                device_type=device_type,
                last_seen=datetime.utcnow(),
                online=True
            )
            db.add(device)
            logger.info(f"[LOGIN] Created new device: {device.device_uuid} ({device.device_name})")
        else:
            # Update existing device with any new information
            device.last_seen = datetime.utcnow()
            device.online = True
            device.device_type = device_type or device.device_type
            
            # If device_uuid was missing, update it
            if not device.device_uuid and device_uuid:
                device.device_uuid = device_uuid
                
        # Commit the device changes
        db.commit()
        db.refresh(device)
        logger.info(f"[LOGIN] Device saved to database: {device.deviceId}")
        
    except Exception as e:
        db.rollback()
        logger.error(f"[LOGIN] Error updating device: {str(e)}", exc_info=True)
        # Don't fail the login if device tracking fails
        device = None
        device.device_name = device_name or device.device_name
        logger.info(f"[LOGIN] Updated existing device: {device.device_uuid} (ID: {device.deviceId})")

    try:
        db.commit()
        db.refresh(device)
        logger.info(f"[LOGIN] Device saved to database: {device.deviceId}")
    except Exception as e:
        db.rollback()
        logger.error(f"[LOGIN] Failed to save device: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to register device")

    # ------------------------------------------------------------------
    # Create an initial per-device tracking file (<device_id>.json)
    # ------------------------------------------------------------------
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"{device.deviceId}_{today}.json"
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
    token_expires_minutes = ACCESS_TOKEN_EXPIRE_MINUTES
    access_token_expires = timedelta(minutes=token_expires_minutes)
    token_data = {"sub": user.username, "user_id": str(user.userId), "device_id": str(device.deviceId) if device else None}
    access_token = create_access_token(
        data=token_data,
        expires_delta=access_token_expires
    )
    
    logger.info(f"[LOGIN] Generated access token for user_id={user.userId}, expires in {token_expires_minutes} minutes")

    # Convert role to string if it's an integer
    role_str = str(user.role) if user.role is not None else "user"
    
    response_data = {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": int(access_token_expires.total_seconds()),
        "user_id": user.userId,
        "username": user.username,
        "role": role_str,  # Ensure role is a string
        "device_id": str(device.deviceId) if device else None,
        "device_uuid": device_uuid,
        "user": {
            "id": str(user.userId),
            "username": user.username,
            "phone_number": user.phone_number,
            "role": role_str  # Ensure role is a string in the nested user object too
        }
    }
    
    logger.info(f"[LOGIN] Login successful for user_id={user.userId}, device_id={device.deviceId if device else 'none'}")
    logger.debug(f"[LOGIN] Response data: {response_data}")
    
    return response_data


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

        # Use user's configured max file size if present, otherwise fall back to 500 MB.
        max_file_size = user.max_file_size or 524_288_000  # 500 MB default
        if file_size > max_file_size:
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
        log_error(f"Upload failed: {str(e)}", e, endpoint="/upload")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {str(e)}"
        )


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
    
    # Try to parse the JSON body
    try:
        body = await request.json()
        if not body:
            return JSONResponse({"error": "Empty request body"}, status_code=400)
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
        
        # Helper function to get or create memory file
        def get_or_create_memory_file(filename, default_content='{}'):
            file = db.query(DBFile).filter(DBFile.userId == user.userId, DBFile.filename == filename).first()
            if not file:
                # Create the file if it doesn't exist
                file = DBFile(
                    userId=user.userId,
                    filename=filename,
                    content=default_content.encode('utf-8'),
                    content_type='application/json',
                    size=len(default_content)
                )
                db.add(file)
                db.commit()
                db.refresh(file)
                logger.info(f"Created new memory file: {filename} for user {user.userId}")
            return file
            
        # Get or create memory files
        try:
            # Long-term memory
            longterm_memory_file = get_or_create_memory_file("long_term_memory.json")
            longterm_content_str = longterm_memory_file.content.decode('utf-8') if longterm_memory_file.content else "{}"
            
            # Short-term memory
            shortterm_memory_file = get_or_create_memory_file("short_term_memory.json")
            shortterm_content_str = shortterm_memory_file.content.decode('utf-8') if shortterm_memory_file.content else "{}"
            
            # Parse JSON content
            longterm_memory_data = json.loads(longterm_content_str) if longterm_content_str.strip() else {}
            shortterm_memory_data = json.loads(shortterm_content_str) if shortterm_content_str.strip() else {}
            
            # Initialize memory managers
            long_term_memory = LongTermMemoryManager(memory_content=longterm_memory_data)
            short_term_memory = ShortTermMemoryManager(memory_content=shortterm_memory_data)
            
        except Exception as e:
            logger.error(f"Error initializing memory managers for user {user.userId}: {str(e)}")
            # Fallback to empty memories
            long_term_memory = LongTermMemoryManager()
            short_term_memory = ShortTermMemoryManager()
            logger.info("Initialized empty memory managers due to error")

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
                notes_parts.append(f"<BEGIN NOTE>{nf.filename}:{note_text}<END NOTE>")
            if notes_parts:
                concatenated_notes = "_".join(notes_parts)
                user_query += "\n\nThese are the notes of the user. Notes are separated by _ and each note is in the format <BEGIN NOTE>filename:note<END NOTE> ::" 
                user_query += concatenated_notes
        except Exception as e:
            logger.error(f"Error loading user notes for query: {e}")

        # Call your AI agent with try/except to handle Vercel environment limitations
        try:
            try:
                
                logger.info(f"Sending query to AI: {user_query[:100]}...")
                # Send query to AI agent using query_openai instead of ask_ai
                response = query_openai(
                    query=user_query,
                    long_term_memory=long_term_memory,
                    short_term_memory=short_term_memory,
                    max_tokens=10000,  # Default max tokens
                    temperature=0.7  # Default temperature
                )
                logger.info("Successfully received response from AI")
            except Exception as e:
                logger.error(f"Error during AI query processing: {str(e)}", exc_info=True)
                raise
            
            # If the query was successful, update the memory
            if not response.startswith("Error:"):
                # Update shortterm memory
                conversations = shortterm_memory_data.get("conversations", [])
                
                summary = ai_query_handler.summarize_conversation(user_query, response)
                updated = ai_query_handler.update_memory(user_query, response, long_term_memory) 
                
                # Update conversations
                conversations += [{
                    "query": user_query, 
                    "response": response, 
                    "summary": summary
                }]
                # limit conversations to 50
                if len(conversations) > 50:
                    conversations = conversations[-50:]
                shortterm_memory_data["conversations"] = conversations

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
            raise file_err
        
        # Log the response
        log_ai_response(response, "/query")
        
        # Update the query record with the response
        db_query.response = response
        db.commit()
        
        # Get the model name from the environment or use a default
        model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        # Estimate tokens used (this is a rough estimate since we don't have exact token count)
        tokens_used = len(response.split()) + len(user_query.split())
        
        return {
            "response": response,
            "query": user_query,
            "chat_id": chat_id,
            "queryId": db_query.queryId,
            "tokens_used": tokens_used,
            "model": model_name
        }
    except Exception as e:
        db.rollback()  # Rollback transaction on error
        error_msg = f"AI query failed: {str(e)}. Type: {type(e).__name__}. Args: {e.args}"
        log_error(error_msg, exc=e, endpoint="/query")
        
        # Include more detailed error information in development
        detail_msg = f"Error processing query: {str(e)}"
        if not os.environ.get("PRODUCTION"):
            detail_msg += f"\nType: {type(e).__name__}\nArgs: {e.args}"
            import traceback
            detail_msg += f"\n\nTraceback:\n{traceback.format_exc()}"
            
        # Always return JSON response
        return JSONResponse(
            status_code=500,
            content={
                "error": detail_msg,
                "type": type(e).__name__,
                "args": str(e.args)
            }
        )
        


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
    request: Request,
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
    endpoint = "/device/heartbeat"
    logger.info(f"[HEARTBEAT] Received heartbeat from user_id={user.userId}, device_id={device_id}")
    
    # Validate device_id
    if not device_id:
        error_msg = "device_id is required for heartbeat. Please include a valid device identifier."
        logger.error(f"[HEARTBEAT] {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Missing device_id",
                "message": error_msg,
                "hint": "Make sure to include a valid device_id parameter with your request. This should be a stable, unique identifier for the device."
            }
        )
    
    # Validate UUID format
    try:
        uuid.UUID(device_id)
    except ValueError:
        error_msg = f"Invalid device_id format. Expected a valid UUID, got: {device_id}"
        logger.error(f"[HEARTBEAT] {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Invalid device_id format",
                "message": error_msg,
                "hint": "The device_id must be a valid UUID (e.g., '123e4567-e89b-12d3-a456-426614174000')"
            }
        )
    
    # Resolve device record prioritising device_uuid when supplied
    device = None
    logger.info(f"[HEARTBEAT] Looking up device with user_id={user.userId}, device_uuid={device_id}")
    
    # First try exact match (for backward compatibility)
    device = db.query(Device).filter(
        Device.userId == user.userId,
        Device.device_uuid == device_id
    ).first()
    
    # If not found and device_id starts with 'chrome-', try without the prefix
    if not device and device_id and device_id.startswith('chrome-'):
        clean_device_id = device_id[7:]  # Remove 'chrome-' prefix
        logger.info(f"[HEARTBEAT] Trying with cleaned device_id: {clean_device_id}")
        device = db.query(Device).filter(
            Device.userId == user.userId,
            Device.device_uuid == clean_device_id
        ).first()
        
        # If found with cleaned ID, update device_uuid to include the prefix for future requests
        if device:
            logger.info(f"[HEARTBEAT] Found device with cleaned ID, updating device_uuid to include prefix")
            device.device_uuid = device_id  # Update to include the prefix
            db.commit()
    
    if not device:
        logger.info(f"[HEARTBEAT] No existing device found for user_id={user.userId} with device_uuid={device_id}, creating new device")
        
        try:
            # Create a new device record
            device = Device(
                userId=user.userId,
                device_uuid=device_id,
                device_name=device_name or f"Device_{device_id[:8]}",
                device_type=device_type or "browser_extension",
                last_seen=now,
                online=True
            )
            db.add(device)
            db.commit()
            db.refresh(device)
            
            logger.info(f"[HEARTBEAT] Created new device: id={device.deviceId}, uuid={device.device_uuid}, name={device.device_name}")
            
        except Exception as e:
            db.rollback()
            error_msg = f"Failed to create new device: {str(e)}"
            logger.error(f"[HEARTBEAT] {error_msg}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Device creation failed",
                    "message": error_msg
                }
            )
    else:
        logger.info(f"[HEARTBEAT] Found device: device_id={device.deviceId}, name={device.device_name}, last_seen={device.last_seen}")

    # Update the last-seen timestamp
    device.last_seen = now
    device.online = True
    device.device_name = device_name or device.device_name
    device.device_type = device_type or device.device_type
    db.commit()
    
    logger.info(f"[HEARTBEAT] Updated device {device.deviceId} - last_seen={now}, online=True")

    # ------------------------------------------------------------------
    # Persist foreground context to the <device_id>_<timestamp>.json file
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d")
    filename = f"{device.deviceId}_{timestamp}.json"
    file_entry = db.query(DBFile).filter(
        DBFile.userId == user.userId,
        DBFile.filename == filename
    ).first()

    if file_entry and file_entry.content:
        try:
            data = json.loads(file_entry.content.decode("utf-8"))
        except Exception:
            data = {
                "deviceId": device.deviceId,
                "device_name": device_name,
                "device_type": device_type,
                "events": []
            }
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

    # remove old device files (10 days old)
    old_files = db.query(DBFile).filter(
        DBFile.userId == user.userId,
        DBFile.filename.like(f"{device.deviceId}_%")
    ).filter(
        DBFile.uploaded_at < (now - timedelta(days=10))
    ).all()
    for old_file in old_files:
        db.delete(old_file)



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
        device.online = False
        device.last_seen = datetime.utcnow()
        logger.info(f"[device_logout] Set device '{device.device_name}' offline")
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
            # Determine online status based on last_seen timestamp rather than the persisted boolean
            online = bool(device.last_seen and device.last_seen >= threshold)
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
    From: str = Form(None), 
    Body: str = Form(""),
    db: Session = Depends(get_db)
):
    """Handle incoming SMS messages from Twilio and forward them to the query endpoint.
    
    Args:
        request: The incoming HTTP request
        From: The sender's phone number (from Twilio form data)
        Body: The message body (from Twilio form data)
        db: Database session
    """
    # If From/Body not provided as parameters, try to get from form data
    form_data = await request.form()
    from_number = From or form_data.get('From', '')
    body = Body or form_data.get('Body', '').strip()
    
    endpoint_name = "/api/webhooks/twilio/incoming-message"
    log_request_start(endpoint_name, "POST", dict(request.headers), request.client.host if request.client else "unknown")
    
    # Log the incoming message
    logger.info(f"[{endpoint_name}] Incoming message from {from_number}: {body}")
    
    try:
        # Normalize phone number (remove non-digits)
        normalized_from = re.sub(r'\D', '', from_number)
        
        # Find user by phone number
        user = db.query(User).filter(User.phone_number == int(normalized_from)).first() if normalized_from.isdigit() else None
        
        if not user:
            logger.warning(f"[{endpoint_name}] Unauthorized access attempt from {from_number}")
            return Response(
                content="<Response><Message>Unauthorized. Please register first.</Message></Response>",
                media_type="application/xml",
                status_code=200
            )
        
        # Create a chat ID for this conversation
        chat_id = f"sms_{normalized_from}"
        user_query_text = body
        
        # Log the query
        logger.info(f"[{endpoint_name}] Processing query from user {user.userId}: {user_query_text}")
        
        # Save the query to database
        try:
            db_query = Query(
                userId=user.userId,
                chatId=chat_id,
                query_text=user_query_text,
                response=None  # Will be updated with AI response
            )
            db.add(db_query)
            db.commit()
            db.refresh(db_query)
            logger.info(f"[{endpoint_name}] Successfully created query with ID: {db_query.queryId}")
            log_ai_call(user_query_text, "default_sms_model", endpoint_name)
        except Exception as e:
            logger.error(f"[{endpoint_name}] Error creating query record: {str(e)}", exc_info=True)
            raise

        # Load conversation history
        conversation_history = []
        try:
            # Get recent messages for context
            recent_messages = db.query(Query).filter(
                Query.userId == user.userId,
                Query.chatId == chat_id
            ).order_by(Query.created_at.desc()).limit(5).all()
            
            # Reverse to get chronological order
            for msg in reversed(recent_messages):
                if msg.query_text:
                    conversation_history.append({"role": "user", "content": msg.query_text})
                if msg.response:
                    conversation_history.append({"role": "assistant", "content": msg.response})
            
            logger.info(f"[{endpoint_name}] Loaded {len(conversation_history)} messages from history")
        except Exception as e:
            logger.error(f"[{endpoint_name}] Error loading conversation history: {str(e)}")
            conversation_history = []
        
        # Prepare the query for the /query endpoint
        query_data = QueryRequest(
            query=user_query_text,
            chat_id=chat_id,
            conversation_history=conversation_history,
            user_id=user.userId
        )
        
        # Call the query endpoint
        try:
            logger.info(f"[{endpoint_name}] Forwarding query to /query endpoint")
            # Call the query endpoint with the request object
            ai_response = await query_endpoint(
                query_data=query_data,
                request=request,
                user=user,
                db=db
            )
            
            response_text = ai_response.response
            logger.info(f"[{endpoint_name}] Received response from /query endpoint")
            
            # Update the query with the response
            db_query.response = response_text
            db.commit()
            
            # Format TwiML response
            twiml_response = f"""
            <Response>
                <Message>{response_text}</Message>
            </Response>
            """
            
            log_response(200, "TwiML reply sent", endpoint_name)
            return Response(
                content=twiml_response.strip(),
                media_type="application/xml",
                status_code=200
            )

        except Exception as e:
            logger.error(f"[{endpoint_name}] Error calling /query endpoint: {str(e)}", exc_info=True)
            db.rollback()
            twiml_error_reply = "<Response><Message>Sorry, an internal error occurred while processing your message.</Message></Response>"
            return Response(content=twiml_error_reply, media_type="application/xml", status_code=500)

    except Exception as e:
        #logger.error(f"[{endpoint_name}] Error processing AI query for SMS: {e}", exc_info=True)
        db.rollback() # Rollback any partial DB changes on error
        twiml_error_reply = "<Response><Message>Sorry, an internal error occurred while processing your message.</Message></Response>"
        return Response(content=twiml_error_reply, media_type="application/xml", status_code=500)


@router.post("/twilio/message-status")
async def handle_twilio_message_status(
    request: Request, 
    MessageSid: str = Form(...), 
    MessageStatus: str = Form(...),
    To: str = Form(None),
    From: str = Form(None),
    ErrorCode: str = Form(None)
):
    """Handle message status callbacks from Twilio.
    
    Args:
        request: The incoming HTTP request
        MessageSid: The unique ID of the message
        MessageStatus: The delivery status of the message
        To: The recipient phone number (optional)
        From: The sender phone number (optional)
        ErrorCode: Error code if message failed (optional)
    """
    endpoint = "/twilio/status"
    client_host = request.client.host if request.client else "unknown_client"
    log_request_start(endpoint, "POST", dict(request.headers), client_host)
    
    # Log the status update
    logger.info(f"Twilio Message Status Update - SID: {MessageSid}, Status: {MessageStatus}")
    
    # If there's an error code, log it as an error
    if ErrorCode:
        logger.error(f"Message {MessageSid} failed with error code: {ErrorCode}")
    
    log_response(200, "OK", "/twilio/status")
    return {"status": "ok"}



@router.post("/webhooks/twilio/incoming-call")
async def handle_twilio_incoming_call(
    request: Request,
    From: str = Form(None),
    SpeechResult: str = Form(None),  # Speech-to-text result from Twilio
    RecordingUrl: str = Form(None),  # URL of the recording if available
    db: Session = Depends(get_db)
):
    """Handle incoming calls from Twilio and respond with AI-generated speech.
    
    Args:
        request: The incoming HTTP request
        From: The caller's phone number (from Twilio form data)
        SpeechResult: The transcribed text from the user's speech (if available)
        RecordingUrl: URL of the recording (if available)
        db: Database session
    """
    endpoint_name = "/webhooks/twilio/incoming-call"
    log_request_start(endpoint_name, "POST", dict(request.headers), request.client.host if request.client else "unknown")
    
    try:
        # Normalize phone number (remove non-digits)
        normalized_from = re.sub(r'\D', '', From) if From else ''
        
        # Find user by phone number
        user = db.query(User).filter(User.phone_number == int(normalized_from)).first() if normalized_from and normalized_from.isdigit() else None
        
        # Create a voice response
        resp = VoiceResponse()
        
        if not user:
            logger.warning(f"[{endpoint_name}] Unauthorized access attempt from {From}")
            resp.say("Sorry, you are not authorized to use this service. Please contact support for assistance.")
            log_response(200, "Unauthorized response sent", endpoint_name)
            return Response(content=str(resp), media_type="application/xml", status_code=200)
        
        # If this is a transcription callback with speech result
        if SpeechResult:
            logger.info(f"[{endpoint_name}] Processing speech input from user {user.userId}: {SpeechResult}")
            
            # Create a chat ID for this conversation
            chat_id = f"call_{normalized_from}"
            
            # Save the query to database
            try:
                db_query = Query(
                    userId=user.userId,
                    chatId=chat_id,
                    query_text=SpeechResult,
                    response=None  # Will be updated with AI response
                )
                db.add(db_query)
                db.commit()
                db.refresh(db_query)
                logger.info(f"[{endpoint_name}] Created query with ID: {db_query.queryId}")
            except Exception as e:
                logger.error(f"[{endpoint_name}] Error creating query record: {str(e)}", exc_info=True)
                db.rollback()
                resp.say("Sorry, there was an error processing your request.")
                return Response(content=str(resp), media_type="application/xml", status_code=200)
            
            # Load conversation history
            conversation_history = []
            try:
                # Get recent messages for context
                recent_messages = db.query(Query).filter(
                    Query.userId == user.userId,
                    Query.chatId == chat_id
                ).order_by(Query.created_at.desc()).limit(5).all()
                
                # Reverse to get chronological order
                for msg in reversed(recent_messages):
                    if msg.query_text:
                        conversation_history.append({"role": "user", "content": msg.query_text})
                    if msg.response:
                        conversation_history.append({"role": "assistant", "content": msg.response})
                
                logger.info(f"[{endpoint_name}] Loaded {len(conversation_history)} messages from history")
            except Exception as e:
                logger.error(f"[{endpoint_name}] Error loading conversation history: {str(e)}")
                conversation_history = []
            
            # Prepare the query for the /query endpoint
            query_data = QueryRequest(
                query=SpeechResult,
                chat_id=chat_id,
                conversation_history=conversation_history,
                user_id=user.userId
            )
            
            try:
                # Call the query endpoint
                logger.info(f"[{endpoint_name}] Forwarding query to /query endpoint")
                ai_response = await query_endpoint(
                    query_data=query_data,
                    user=user,
                    db=db
                )
                
                response_text = ai_response.response
                logger.info(f"[{endpoint_name}] Received response from /query endpoint")
                
                # Update the query with the response
                db_query.response = response_text
                db.commit()
                
                # Speak the response to the user
                resp.say(response_text)
                
                # Ask if there's anything else we can help with
                resp.say("Is there anything else I can help you with?")
                
                # Record the user's response
                resp.record(
                    action=f"/webhooks/twilio/incoming-call?From={From}",
                    method="POST",
                    finish_on_key="#",
                    transcribe=True,
                    transcribe_callback=f"/webhooks/twilio/incoming-call?From={From}"
                )
                
                log_response(200, "AI response spoken to user", endpoint_name)
                return Response(content=str(resp), media_type="application/xml", status_code=200)
                
            except Exception as e:
                logger.error(f"[{endpoint_name}] Error calling /query endpoint: {str(e)}", exc_info=True)
                db.rollback()
                resp.say("I'm sorry, I encountered an error processing your request. Please try again later.")
                return Response(content=str(resp), media_type="application/xml", status_code=200)
        
        # Initial greeting for new calls
        logger.info(f"[{endpoint_name}] New call from user {user.userId} ({user.username})")
        resp.say(f"Hello {user.username}, this is Gad's assistant. How can I help you today?")
        
        # Record the user's message
        resp.record(
            action=f"/webhooks/twilio/incoming-call?From={From}",
            method="POST",
            finish_on_key="#",
            transcribe=True,
            transcribe_callback=f"/webhooks/twilio/incoming-call?From={From}"
        )
        
        log_response(200, "Initial greeting sent", endpoint_name)
        return Response(content=str(resp), media_type="application/xml", status_code=200)
        
    except Exception as e:
        logger.error(f"[{endpoint_name}] Unexpected error: {str(e)}", exc_info=True)
        resp = VoiceResponse()
        resp.say("I'm sorry, an unexpected error occurred. Please try your call again later.")
        return Response(content=str(resp), media_type="application/xml", status_code=200)


@router.post("/webhooks/twilio/transcription-callback")
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
