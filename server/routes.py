"""API Routes Module for LMS Platform.

This module defines all the API endpoints for the LMS platform, including:
- User authentication and management
- File operations (upload, download, listing, deletion)
- AI query handling
- URL tracking
- User profile information
"""

import os
import json
import logging
import re
import uuid
import atexit
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import unquote, quote
import mimetypes

from fastapi import APIRouter, Depends, HTTPException, UploadFile as FastAPIFile, File, Request, Header, Form, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse, JSONResponse, Response, FileResponse
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

from .utils import compute_sha256
from .db import User, File as DBFile, Query, Device, Session, SessionLocal
from .schemas import RegisterRequest, UserResponse
from .auth import (
    get_db, get_password_hash, authenticate_user, create_access_token,
    get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES
)
from aiagent.handler import query as ai_query_handler
from aiagent.memory.memory_manager import LongTermMemoryManager, ShortTermMemoryManager
from aiagent.context.reference import read_references
# from aiagent.tools.voice import Whisper, download_audio

# whisper_transcriber = Whisper(model_size='tiny')  # Load once at the top


# def transcribe_audio(audio_url: str) -> str:
#     """Transcribe audio from a URL using the Whisper model.
    
#     Args:
#         audio_url: URL of the audio file to transcribe
        
#     Returns:
#         str: The transcribed text
        
#     Raises:
#         Exception: If transcription fails
#     """
#     logger.info(f"Transcribing audio from URL: {audio_url}")
#     try:
#         # Download the audio file
#         audio_path = download_audio(audio_url)
        
#         # Transcribe the audio
#         text, language = whisper_transcriber.transcribe(audio_path)
#         logger.info(f"Transcription completed. Detected language: {language}")
        
#         # Clean up the downloaded file
#         try:
#             os.remove(audio_path)
#         except Exception as e:
#             logger.warning(f"Failed to delete temporary audio file {audio_path}: {str(e)}")
            
#         return text
        
#     except Exception as e:
#         logger.error(f"Error in transcribe_audio: {str(e)}")
#         raise Exception(f"Failed to transcribe audio: {str(e)}")

# Set up logger first
logger = logging.getLogger("lms.server")
logger.setLevel(logging.INFO)

# Configure logging formatter
formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')

# Always log to console (stdout), which Vercel captures
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Only log to file if not running on Vercel (Vercel sets the VERCEL env var)
if not os.environ.get("VERCEL"):
    try:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "lms_server.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"[Logging] Could not set up file logging: {e}")
else:
    logger.warning("[Logging] File logging is disabled (running on Vercel or read-only filesystem)")

# Initialize router
router = APIRouter()

# Global variable to store the status message
status_message = "Thoth API is starting up..."

# Function to update the status message
def update_status():
    global status_message
    try:
        with SessionLocal() as db:
            user_count = db.query(User).count()
            online_device_count = db.query(Device).filter(
                Device.last_seen > datetime.utcnow() - timedelta(minutes=5)
            ).count()
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status_message = (
                f"Hello from Thoth API is running (as of {current_time}). "
                f"Total users: {user_count}. Devices online: {online_device_count}"
            )
    except Exception as e:
        logger.error(f"Error updating status: {e}")


def send_status():
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"Thoth API Status: Running as of {current_time}"
        recipient_phone = "+18073587137"  # Hardcoded E.164 format
        success = send_twilio_message(recipient_phone, message)
        if not success:
            logger.error(f"[send_status] Failed to send SMS to {recipient_phone}")
        else:
            logger.info(f"[send_status] SMS sent to {recipient_phone} at {current_time}")
    except Exception as e:
        logger.error(f"[send_status] Error sending SMS: {e}")


def auto_disconnect_stale_devices():
    try:
        with SessionLocal() as db:
            threshold = datetime.utcnow() - timedelta(minutes=5)
            stale_devices = db.query(Device).filter(
                Device.last_seen < threshold
            ).all()

            for device in stale_devices:
                logger.info(f"Device {device.deviceId} considered stale (last_seen {device.last_seen})")

    except Exception as e:
        logger.error(f"Auto-disconnect failed: {e}")


# Initialize the scheduler
scheduler = None

def start_scheduler():
    global scheduler
    if scheduler is None:
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            update_status,
            trigger=IntervalTrigger(minutes=1),
            id='update_status_job',
            name='Update status every minute',
            replace_existing=True
        )
        scheduler.add_job(
            send_status,
            trigger=IntervalTrigger(minutes=200),
            id='send_status_job',
            name='Send status every 10 minute',
            replace_existing=True
        )
        scheduler.add_job(
            auto_disconnect_stale_devices,
            trigger=IntervalTrigger(minutes=2),
            id='auto_disconnect_job',
            name='Auto disconnect stale devices',
            replace_existing=True
        )
        scheduler.start()
        #logger.info("Scheduler started for status updates")
        atexit.register(lambda: scheduler.shutdown() if scheduler else None)

# Start the scheduler when the module loads
start_scheduler()

# Initialize assets folder
ASSETS_FOLDER = "assets"
os.makedirs(ASSETS_FOLDER, exist_ok=True)

# Logging helpers (stubs for demonstration)
def log_request_start(endpoint, method, headers, client_host):
    logger.info(f"[{endpoint}] {method} request from {client_host}, headers: {headers}")
def log_request_payload(payload, endpoint):
    logger.info(f"[{endpoint}] Payload: {payload}")
def log_validation(field, value, valid, endpoint):
    logger.info(f"[{endpoint}] Validation: {field} valid={valid}")
def log_error(msg, exc=None, context=None, endpoint=None):
    logger.error(f"[{endpoint}] ERROR: {msg} | Context: {context}")
def log_response(status, response, endpoint):
    logger.info(f"[{endpoint}] Responded with status {status}: {response}")

def log_ai_call(query, model, endpoint):
    logger.info(f"[{endpoint}] AI call to {model} with query: {query}")
def log_ai_response(response, endpoint):
    logger.info(f"[{endpoint}] AI response: {response}")
def log_something(something, endpoint):
    logger.info(f"[{endpoint}] Something: {something}")

@router.get("/")
def root():
    """Root endpoint that shows API status and statistics.
    
    Returns:
        dict: API status message including current time and user count
    """
    return {"message": status_message}

@router.get("/favicon.ico")
def favicon():
    favicon_path = os.path.join(os.path.dirname(__file__), "../static/favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return {"detail": "No favicon found"}

@router.post("/register")
def register(req: RegisterRequest, db: SessionLocal = Depends(get_db)):
    """Register a new user in the system.
    
    Args:
        req: The registration request containing username and password
        db: Database session dependency
        
    Returns:
        dict: Registration confirmation with userId
        
    Raises:
        HTTPException: 400 error if username already exists
        HTTPException: 400 error if phone number already exists
    """
    # Check for existing user by username
    existing_user = db.query(User).filter(User.username == req.username).first()
    if existing_user:
        #logger.warning(f"[register] Registration attempt with existing username: {req.username}")
        raise HTTPException(status_code=400, detail="Username already registered")

    # Check for existing user by phone_number if provided
    if req.phone_number is not None:
        existing_user_by_phone = db.query(User).filter(User.phone_number == req.phone_number).first()
        if existing_user_by_phone:
            #logger.warning(f"[register] Registration attempt with existing phone number: {req.phone_number}")
            raise HTTPException(status_code=400, detail="Phone number already registered")

    # Create new user
    hashed_password = get_password_hash(req.password)
    new_user = User(
        username=req.username, 
        hashed_password=hashed_password,
        phone_number=req.phone_number, # Ensure phone_number is assigned here
        role=req.role  # Assign role from request
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    user_folder = os.path.join(ASSETS_FOLDER, str(new_user.userId))
    os.makedirs(user_folder, exist_ok=True)
    
    # Create shortterm and longterm memory files for the user
    shortterm_memory = {"conversations": [], "active": [{"device": "", "path": "", "title": "", "timestamp": ""}]}
    longterm_memory = {
        "userProfile_username": new_user.username,  
        "userProfile_role": new_user.role,
        "userProfile_language": "English",
        "userProfile_age": "25",
        "userProfile_gender": "male",
        "userProfile_country": "Canada",
        "userProfile_city": "London",
        "userProfile_address": "Unknown",
        "userProfile_postal_code": "Unknown",
        "userProfile_email": "Unknown",
        "userProfile_phone_number": new_user.phone_number, 
        "userProfile_occupation": "Unknown",
        "userProfile_marital_status": "Unknown",
        "userProfile_children": "Unknown",
        "userProfile_income": "Unknown",
        "userProfile_education": "Unknown",
        "userProfile_employment": "Unknown",

    }
    
    # Save memory files to the database
    shortterm_memory_file = DBFile(
        filename="short_term_memory.json",
        userId=new_user.userId,
        size=len(json.dumps(shortterm_memory)),
        content=json.dumps(shortterm_memory).encode('utf-8'),
        content_type="application/json"
    )
    
    longterm_memory_file = DBFile(
        filename="long_term_memory.json",
        userId=new_user.userId,
        size=len(json.dumps(longterm_memory)),
        content=json.dumps(longterm_memory).encode('utf-8'),
        content_type="application/json"
    )
    
    db.add(shortterm_memory_file)
    db.add(longterm_memory_file)
    db.commit()
    
    return {"message": "Registered successfully", "userId": new_user.userId}

@router.delete("/user/{username}")
def delete_user(username: str, current_user: User = Depends(get_current_user), db: SessionLocal = Depends(get_db)):
    """Delete a user account.
    
    Users can only delete their own accounts, not others.
    
    Args:
        username: The username of the account to delete
        current_user: The authenticated user making the request
        db: Database session dependency
        
    Returns:
        dict: Confirmation message
        
    Raises:
        HTTPException: 403 if trying to delete another user's account
        HTTPException: 404 if user not found
    """
    # Verify user can only delete their own account
    if current_user.username != username:
        raise HTTPException(status_code=403, detail="You can only delete your own account.")

    # Find the user to delete
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        # Delete user-related files (skip if in serverless environment)
        if not os.environ.get("VERCEL"):
            try:
                user_folder = os.path.join(ASSETS_FOLDER, str(user.userId))
                if os.path.exists(user_folder):
                    # Delete user directory (optional, can be removed if causing issues)
                    # shutil.rmtree(user_folder)
                    logger.info(f"Deleted user folder: {user_folder}")
            except Exception as e:
                # Just log the error but continue with deletion
                logger.warning(f"Could not delete user files: {e}")

        # Delete the user from the database
        db.delete(user)
        db.commit()

        return {"message": f"User '{username}' deleted successfully."}
    except Exception as e:
        db.rollback()
        #logger.error(f"[delete_user] Exception type: {type(e)}, repr: {repr(e)}")
        #logger.error(f"Error deleting user {username}: {str(e)}")
        if os.environ.get("VERCEL"):
            # In Vercel, return a more user-friendly error only for non-auth errors
            return {"message": f"Partial user deletion for '{username}'. Database records removed but user files may remain due to serverless environment limitations."}
        raise HTTPException(status_code=500, detail=f"Error deleting user: {str(e)}")

@router.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: SessionLocal = Depends(get_db)):
    """Authenticate user and generate access token.
    
    This endpoint conforms to the OAuth2 password flow standard.
    
    Args:
        form_data: OAuth2 form containing username and password
        db: Database session dependency
        
    Returns:
        dict: JWT access token and token type
        
    Raises:
        HTTPException: 401 if authentication fails
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=ACCESS_TOKEN_EXPIRE_MINUTES
    )
    return {"access_token": access_token, "token_type": "bearer"}



@router.post("/upload")
async def upload_file(
    file: FastAPIFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
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



@router.post("/query")
async def queryEndpoint(request: Request, user: User = Depends(get_current_user), db: SessionLocal = Depends(get_db)):
    """Process an AI query from a user.
    
    Sends the query to the OpenAI API and returns the response. The query and response
    are associated with a chat ID for maintaining conversation context.
    All queries and responses are stored in the database for future reference.
    
    Args:
        request: The HTTP request containing the query data
        user: The authenticated user making the query
        db: Database session dependency
        
    Returns:
        dict: The AI response along with query details

    Raises:
        HTTPException: 400 if required parameters are missing
        HTTPException: 500 if OpenAI API call fails
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
    device_name: str = Form(...),
    device_type: str = Form(...),
    current_app: str = Form(None),
    current_page: str = Form(None),
    current_url: str = Form(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    now = datetime.utcnow()

    device = db.query(Device).filter(
        Device.userId == user.userId,
        Device.device_name == device_name
    ).first()

    if not device:
        device = Device(
            userId=user.userId,
            device_name=device_name,
            device_type=device_type
        )
        db.add(device)
        db.commit()
        db.refresh(device)

    device.last_seen = now
    db.commit()

    return {
        "message": "Device heartbeat received",
        "deviceId": device.deviceId,
        "last_seen": device.last_seen.isoformat()
    }


@router.post("/device/logout")
async def device_logout(
    device_name: str = Form(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    device = db.query(Device).filter(
        Device.userId == user.userId,
        Device.device_name == device_name
    ).first()

    if device:
        device.last_seen = datetime.utcnow()
        db.commit()

    return {"status": "logged out", "deviceId": device.deviceId if device else None}


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
async def handle_twilio_incoming_message(request: Request, From: str = Form(...), Body: str = Form(...), db: Session = Depends(get_db)):
    """
    Handles incoming SMS messages from Twilio, processes them via AI, and sends a reply.
    Twilio sends data as 'application/x-www-form-urlencoded'.
    """
    client_host = request.client.host if request.client else "unknown_client"
    endpoint_name = "/api/webhooks/twilio/incoming-message"
    log_request_start(endpoint_name, "POST", dict(request.headers), client_host)
    #logger.info(f"[{endpoint_name}] Twilio Incoming SMS from {From}: {Body}")

    normalized_from_number_str = re.sub(r'\D', '', From)
    user_query_text = Body
    found_user: Optional[User] = None

    if not normalized_from_number_str:
        #logger.warning(f"[{endpoint_name}] Received empty or invalid 'From' number: {From}. Cannot look up user.")
        twiml_response = "<Response><Message>Sorry, we could not identify your phone number.</Message></Response>"
        log_response(200, twiml_response, endpoint_name)
        return Response(content=twiml_response, media_type="application/xml", status_code=200)

    phone_number_to_lookup = None

    try:
        phone_number_to_lookup = int(normalized_from_number_str)
        found_user = db.query(User).filter(User.phone_number == phone_number_to_lookup).first()
        # if found_user:
            #logger.info(f"[{endpoint_name}] Matched Twilio number {normalized_from_number_str} (parsed as {phone_number_to_lookup}) to user {found_user.username} (ID: {found_user.userId}) via direct DB lookup.")
    except ValueError:
        #logger.error(f"[{endpoint_name}] Could not convert normalized phone number '{normalized_from_number_str}' to int for DB lookup.")
        found_user = None # Ensure found_user is None if conversion fails
    except Exception as e:
        #logger.error(f"[{endpoint_name}] Error during direct DB lookup for phone number {normalized_from_number_str}: {e}")
        found_user = None # Ensure found_user is None on other DB errors



    if not found_user:
        #logger.warning(f"[{endpoint_name}] No user found for Twilio number {normalized_from_number_str} ({From}) after all lookups. Sending 'not recognized' message.")
        twiml_response = "<Response><Message>Sorry, you are not recognized by Gad.</Message></Response>"
        log_response(200, twiml_response, endpoint_name)
        return Response(content=twiml_response, media_type="application/xml", status_code=200)

    # User found, proceed with AI query
    user = found_user
    chat_id = f"sms_{normalized_from_number_str}" # Consistent chat ID for SMS user

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
async def handle_twilio_message_status(request: Request, MessageSid: str = Form(...), MessageStatus: str = Form(...)):
    """
    Handles delivery status updates for outbound messages from Twilio.
    """
    client_host = request.client.host if request.client else "unknown_client"
    log_request_start("/api/webhooks/twilio/message-status", "POST", dict(request.headers), client_host)
    #logger.info(f"[/api/webhooks/twilio/message-status] Twilio Message SID {MessageSid} status: {MessageStatus}")
    
    # --- Your logic here to update message status in your DB ---
    
    log_response(200, "OK", "/api/webhooks/twilio/message-status")
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



def send_twilio_message(to_phone_number: str, message: str):
    try:
        # Initialize Twilio client
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        from_phone_number = os.environ.get("TWILIO_FROM_NUMBER")

        if not all([account_sid, auth_token, from_phone_number]):
            #logger.error("[send_twilio_message] Missing Twilio environment variables")
            return False

        client = Client(account_sid, auth_token)
        
        # Send SMS
        client.messages.create(
            body=message,
            from_=from_phone_number,
            to=to_phone_number
        )
        #logger.info(f"[send_twilio_message] Successfully sent SMS to {to_phone_number}: {message}")
        return True
    except TwilioRestException as e:
        #logger.error(f"[send_twilio_message] Twilio error sending SMS to {to_phone_number}: {e}")
        return False
    except Exception as e:
        #logger.error(f"[send_twilio_message] Unexpected error sending SMS to {to_phone_number}: {e}")
        return False
