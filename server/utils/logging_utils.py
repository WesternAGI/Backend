"""
Logging Utilities Module

This module provides specialized logging functions to standardize
and enhance the logging across the entire server application.
"""

import logging
import os
import sys
import platform
import psutil
import json
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, Union
from flask import request

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)

def log_server_lifecycle(event: str, details: Optional[Dict] = None) -> None:
    """Log server lifecycle events with detailed information.
    
    Args:
        event (str): The lifecycle event (startup, shutdown, etc.)
        details (dict, optional): Additional details about the event
    """
    logger.info(f"[SERVER] Lifecycle event: {event}")
    logger.info(f"[SERVER] Timestamp: {datetime.now().isoformat()}")
    if details:
        logger.info(f"[SERVER] Event details: {details}")

def log_server_health() -> None:
    """Log server health metrics including CPU, memory, and threads."""
    process = psutil.Process()
    logger.info(f"[SERVER] CPU usage: {psutil.cpu_percent()}%")
    logger.info(f"[SERVER] Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    logger.info(f"[SERVER] Thread count: {process.num_threads()}")
    logger.info(f"[SERVER] Process start time: {datetime.fromtimestamp(process.create_time()).isoformat()}")
    logger.info(f"[SERVER] System uptime: {datetime.fromtimestamp(psutil.boot_time()).isoformat()}")

def log_request_start(endpoint: str, method: Optional[str] = None, 
                    headers: Optional[Dict] = None, remote_addr: Optional[str] = None) -> None:
    """Log the start of a request with detailed information.
    
    Args:
        endpoint (str): The API endpoint being accessed
        method (str, optional): The HTTP method used
        headers (dict, optional): The request headers
        remote_addr (str, optional): The client's IP address
    """
    method = method or request.method if request else 'UNKNOWN'
    remote_addr = remote_addr or (request.remote_addr if request else 'unknown')
    headers = headers or (dict(request.headers) if request else {})
    
    logger.info(
        "[REQUEST] %s %s from %s\nHeaders: %s",
        method,
        endpoint,
        remote_addr,
        {k: v for k, v in headers.items() if k.lower() not in ['authorization', 'cookie']}
    )

def log_request_payload(payload: Any, endpoint: str) -> None:
    """Log details about the request payload.
    
    Args:
        payload: The request payload
        endpoint: The API endpoint being accessed
    """
    if payload:
        try:
            # Safely convert payload to string, handling non-serializable objects
            if hasattr(payload, 'dict'):
                payload_str = str(payload.dict())
            elif isinstance(payload, (str, bytes)):
                payload_str = str(payload)[:500]  # Truncate long strings
            else:
                payload_str = str(payload)[:500]
                
            logger.debug("[PAYLOAD] %s: %s", endpoint, payload_str)
        except Exception as e:
            logger.warning("[PAYLOAD] Failed to log payload for %s: %s", endpoint, str(e))

def log_validation(field: str, value: Any, valid: bool, endpoint: str) -> None:
    """Log field validation results.
    
    Args:
        field (str): The field being validated
        value (any): The value being validated (will be truncated if string)
        valid (bool): Whether validation passed
        endpoint (str): The API endpoint being accessed
    """
    value_str = str(value)
    if len(value_str) > 100:  # Truncate long values
        value_str = value_str[:100] + "..."
    
    if valid:
        logger.debug("[VALID] %s: %s is valid", field, value_str)
    else:
        logger.warning("[VALID] %s: %s is invalid", field, value_str)

def log_error(msg: str, exc: Optional[Exception] = None, context: Optional[Dict] = None, 
              endpoint: Optional[str] = None) -> None:
    """Log an error with optional exception and context.
    
    Args:
        msg: Error message
        exc: Optional exception that caused the error
        context: Additional context about the error
        endpoint: Optional API endpoint where the error occurred
    """
    log_msg = f"Error in {endpoint}: {msg}" if endpoint else f"Error: {msg}"
    if context:
        log_msg += f"\nContext: {context}"
    
    if exc:
        logger.exception(log_msg, exc_info=exc)
    else:
        logger.error(log_msg)


def log_response(status_code: int, response: Any, endpoint: str) -> None:
    """Log the response being sent back to the client.
    
    Args:
        status_code: HTTP status code
        response: The response data
        endpoint: The API endpoint that was called
    """
    log_level = logging.INFO if status_code < 400 else logging.ERROR
    logger.log(
        log_level,
        "Response from %s - Status: %s, Response: %s",
        endpoint, status_code, response
    )


def log_ai_call(query: str, model: str, endpoint: str) -> None:
    """Log when an AI query is made.
    
    Args:
        query: The user's query
        model: The AI model being used
        endpoint: The API endpoint handling the query
    """
    logger.info(
        "AI Query - Endpoint: %s, Model: %s, Query: %s",
        endpoint, model, query
    )


def log_ai_response(response: str, endpoint: str) -> None:
    """Log the AI's response.
    
    Args:
        response: The AI's response
        endpoint: The API endpoint that handled the query
    """
    logger.debug("AI Response from %s: %s", endpoint, response)


def log_something(something: Any, endpoint: str) -> None:
    """Generic logging function for miscellaneous information.
    
    Args:
        something: The thing to log
        endpoint: The API endpoint where this is being logged from
    """
    logger.info("Log from %s: %s", endpoint, something)
