"""Service layer for handling business logic and utility functions.

This module contains non-endpoint functions that handle core business logic,
background tasks, and utility functions used across the application.
"""

import os
import logging
import json
import uuid
import mimetypes
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from urllib.parse import unquote, quote

# FastAPI
from fastapi import HTTPException, status, UploadFile
from fastapi.responses import FileResponse

# Scheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Twilio
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

# Application imports
from .db import SessionLocal, Device, User, File as DBFile, Query, Session
from .config import settings
from .utils import (
    log_something,
    log_error,
    log_server_health,
    log_server_lifecycle
)
from .utils import compute_sha256
from .auth import get_password_hash, create_access_token

# AI Agent imports
from aiagent.handler import query as ai_query_handler
from aiagent.memory.memory_manager import LongTermMemoryManager, ShortTermMemoryManager
from aiagent.context.reference import read_references

# Initialize logger
logger = logging.getLogger(__name__)


# Initialize scheduler
scheduler = None

# Initialize assets folder
ASSETS_FOLDER = "assets"
os.makedirs(ASSETS_FOLDER, exist_ok=True)


def get_status_message() -> str:
    """Generate a status message with current time and user count.
    
    Returns:
        str: Status message
    """
    db = SessionLocal()
    try:
        user_count = db.query(User).count()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Thoth API is running. Users: {user_count}, Time: {current_time}"
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return f"Thoth API is running. Error getting status: {e}"
    finally:
        db.close()


def update_status():
    """Update the last seen timestamp of all devices."""
    db = SessionLocal()
    try:
        # Update last_seen timestamp for all devices
        db.query(Device).update({Device.last_seen: datetime.utcnow()})
        db.commit()
        logger.info("Updated device timestamps")
    except Exception as e:
        logger.error(f"Error updating device timestamps: {e}")
        db.rollback()
    finally:
        db.close()


def send_status():
    """Send status update to all connected devices."""
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"Thoth API Status: Running as of {current_time}"
        recipient_phone = "+18073587137"  # Hardcoded E.164 format
        success = send_twilio_message(recipient_phone, message)
        if not success:
            log_error(f"[send_status] Failed to send SMS to {recipient_phone}")
        else:
            log_something(f"[send_status] SMS sent to {recipient_phone} at {current_time}")
    except Exception as e:
        log_error(f"[send_status] Error sending SMS: {e}")


def auto_disconnect_stale_devices():
    """Log stale devices that haven't sent a heartbeat recently."""
    db = SessionLocal()
    try:
        # Log devices that haven't been seen in the last 5 minutes
        stale_time = datetime.utcnow() - timedelta(minutes=5)
        stale_devices = db.query(Device).filter(Device.last_seen < stale_time).all()
        
        for device in stale_devices:
            logger.info(f"Device {device.deviceId} (UUID: {device.device_uuid}) last seen at {device.last_seen} is stale")
        
        logger.info(f"Found {len(stale_devices)} stale devices")
    except Exception as e:
        logger.error(f"Error checking for stale devices: {e}")
        db.rollback()
    finally:
        db.close()


def start_scheduler():
    """Start the background scheduler for periodic tasks."""
    global scheduler
    
    if scheduler is not None:
        logger.warning("Scheduler already running")
        return
    
    try:
        scheduler = BackgroundScheduler()
        
        # Add jobs
        scheduler.add_job(
            update_status,
            trigger=IntervalTrigger(minutes=1),
            id='update_status_job',
            name='Update status every minute',
            replace_existing=True
        )
        
        scheduler.add_job(
            send_status,
            trigger=IntervalTrigger(minutes=200),  # ~3.3 hours
            id='send_status_job',
            name='Send status periodically',
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
        logger.info("Scheduler started successfully")
        
        # Register shutdown handler
        import atexit
        atexit.register(lambda: scheduler.shutdown() if scheduler else None)
        
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
        if scheduler is not None:
            scheduler.shutdown()
            scheduler = None


def send_twilio_message(to_phone_number: str, message: str) -> Dict[str, Any]:
    """Send an SMS message using Twilio.
    
    Args:
        to_phone_number: Recipient's phone number in E.164 format
        message: The message content to send
        
    Returns:
        Dict containing the message SID and status if successful
        
    Raises:
        HTTPException: If there's an error sending the message
    """
    try:
        # Initialize Twilio client
        client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
        
        # Send message
        twilio_message = client.messages.create(
            body=message,
            from_=settings.twilio_phone_number,
            to=to_phone_number
        )
        
        return {
            "message_sid": twilio_message.sid,
            "status": twilio_message.status,
            "to": twilio_message.to,
            "date_created": twilio_message.date_created.isoformat() if twilio_message.date_created else None
        }
        
    except TwilioRestException as e:
        logger.error(f"Twilio API error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send message: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error sending Twilio message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while sending the message"
        )
