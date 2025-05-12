#!/usr/bin/env python3

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
from flask import request

logger = logging.getLogger(__name__)

def log_server_lifecycle(event, details=None):
    """Log server lifecycle events with detailed information.
    
    Args:
        event (str): The lifecycle event (startup, shutdown, etc.)
        details (dict, optional): Additional details about the event
    """
    logger.info(f"[SERVER] Lifecycle event: {event}")
    logger.info(f"[SERVER] Timestamp: {datetime.now().isoformat()}")
    if details:
        logger.info(f"[SERVER] Event details: {details}")

def log_server_health():
    """Log server health metrics including CPU, memory, and threads."""
    process = psutil.Process()
    logger.info(f"[SERVER] CPU usage: {psutil.cpu_percent()}%")
    logger.info(f"[SERVER] Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    logger.info(f"[SERVER] Thread count: {process.num_threads()}")
    logger.info(f"[SERVER] Process start time: {datetime.fromtimestamp(process.create_time()).isoformat()}")
    logger.info(f"[SERVER] System uptime: {datetime.fromtimestamp(psutil.boot_time()).isoformat()}")

def log_request_start(endpoint, method=None, headers=None, remote_addr=None):
    """Log the start of a request with detailed information.
    
    Args:
        endpoint (str): The API endpoint being accessed
        method (str, optional): The HTTP method used
        headers (dict, optional): The request headers
        remote_addr (str, optional): The client's IP address
    """
    if not method and request:
        method = request.method
    if not headers and request:
        headers = request.headers
    if not remote_addr and request:
        remote_addr = request.remote_addr
    
    logger.info(f"[SERVER] Request started for {endpoint}")
    logger.info(f"[SERVER] Method: {method}")
    logger.info(f"[SERVER] Remote address: {remote_addr}")
    logger.info(f"[SERVER] Headers count: {len(headers) if headers else 0}")
    logger.debug(f"[SERVER] Headers: {dict(headers) if headers else {}}")
    logger.info(f"[SERVER] Timestamp: {datetime.now().isoformat()}")
    logger.info(f"[SERVER] Request path: {request.path if request else endpoint}")

def log_request_payload(payload, endpoint):
    """Log details about the request payload.
    
    Args:
        payload (dict): The request payload
        endpoint (str): The API endpoint being accessed
    """
    if payload:
        try:
            payload_size = len(json.dumps(payload))
            payload_keys = list(payload.keys()) if isinstance(payload, dict) else "non-dict"
            logger.info(f"[SERVER] Payload size for {endpoint}: {payload_size} bytes")
            logger.info(f"[SERVER] Payload keys: {payload_keys}")
            logger.info(f"[SERVER] Payload type: {type(payload).__name__}")
            
            # Truncate large payloads for debug logging
            if isinstance(payload, dict):
                truncated_payload = {}
                for k, v in payload.items():
                    if isinstance(v, str) and len(v) > 100:
                        truncated_payload[k] = v[:100] + "..."
                    else:
                        truncated_payload[k] = v
                logger.debug(f"[SERVER] Payload content (truncated): {truncated_payload}")
            else:
                logger.debug(f"[SERVER] Payload content (truncated): {str(payload)[:100]}...")
        except Exception as e:
            logger.error(f"[SERVER] Error logging payload: {str(e)}")
    else:
        logger.info(f"[SERVER] No payload for {endpoint}")

def log_validation(field, value, valid, endpoint):
    """Log validation of request fields.
    
    Args:
        field (str): The field being validated
        value (any): The value being validated (will be truncated if string)
        valid (bool): Whether validation passed
        endpoint (str): The API endpoint being accessed
    """
    # Truncate value if it's a string
    if isinstance(value, str):
        displayed_value = value[:30] + "..." if len(value) > 30 else value
    else:
        displayed_value = value
        
    logger.info(f"[SERVER] Validating {field} for {endpoint}: {displayed_value}")
    logger.info(f"[SERVER] Validation result: {'valid' if valid else 'invalid'}")
    logger.info(f"[SERVER] Field type: {type(value).__name__}")
    if isinstance(value, str):
        logger.info(f"[SERVER] Field length: {len(value)}")

def log_ai_call(query, model, endpoint):
    """Log AI API calls with query information.
    
    Args:
        query (str): The query sent to the AI
        model (str): The AI model being used
        endpoint (str): The API endpoint triggering the AI call
    """
    logger.info(f"[SERVER] AI call for {endpoint}")
    logger.info(f"[SERVER] Model: {model}")
    logger.info(f"[SERVER] Query length: {len(query)}")
    logger.info(f"[SERVER] AI call timestamp: {datetime.now().isoformat()}")
    logger.debug(f"[SERVER] Query snippet: {query[:50]}...")

def log_ai_response(response, endpoint):
    """Log AI response information.
    
    Args:
        response (str): The response from the AI
        endpoint (str): The API endpoint that triggered the AI call
    """
    length = len(response) if response else 0
    logger.info(f"[SERVER] AI response for {endpoint}")
    logger.info(f"[SERVER] Response length: {length}")
    logger.info(f"[SERVER] Response type: {type(response).__name__}")
    logger.info(f"[SERVER] Response timestamp: {datetime.now().isoformat()}")
    if response:
        logger.debug(f"[SERVER] Response snippet: {response[:50]}...")

def log_file_operation(action, path, success, endpoint):
    """Log file operations with details.
    
    Args:
        action (str): The file operation (read, write, etc.)
        path (str): The file path
        success (bool): Whether the operation succeeded
        endpoint (str): The API endpoint triggering the file operation
    """
    logger.info(f"[SERVER] File {action} for {endpoint}: {path}")
    logger.info(f"[SERVER] Operation success: {success}")
    if os.path.exists(path):
        logger.info(f"[SERVER] File size: {os.path.getsize(path)} bytes")
        logger.info(f"[SERVER] File modified: {datetime.fromtimestamp(os.path.getmtime(path)).isoformat()}")
    logger.info(f"[SERVER] Operation timestamp: {datetime.now().isoformat()}")

def log_response(status, body, endpoint):
    """Log details about the response being sent.
    
    Args:
        status (int): The HTTP status code
        body (any): The response body (will be truncated)
        endpoint (str): The API endpoint
    """
    logger.info(f"[SERVER] Response for {endpoint} status: {status}")
    if body:
        body_str = str(body)
        body_size = len(body_str)
        logger.info(f"[SERVER] Response size: {body_size}")
        logger.debug(f"[SERVER] Response body snippet: {body_str[:50]}...")
    else:
        logger.info(f"[SERVER] Empty response body")
    logger.info(f"[SERVER] Response timestamp: {datetime.now().isoformat()}")

def log_error(message, exception=None, context=None, endpoint=None):
    """Log detailed error information.
    
    Args:
        message (str): The error message
        exception (Exception, optional): The exception object
        context (dict, optional): Additional context about the error
        endpoint (str, optional): The API endpoint where the error occurred
    """
    stack_trace = traceback.format_exc() if exception else "No stack trace"
    endpoint_info = endpoint or (request.path if request else "unknown")
    
    logger.error(f"[SERVER] Error in {endpoint_info}: {message}")
    logger.error(f"[SERVER] Stack trace: {stack_trace}")
    if context:
        logger.error(f"[SERVER] Error context: {context}")
    logger.error(f"[SERVER] Error timestamp: {datetime.now().isoformat()}")
    
    if request:
        try:
            logger.error(f"[SERVER] Request headers: {dict(request.headers)}")
            logger.error(f"[SERVER] Request method: {request.method}")
            payload = request.get_json(silent=True)
            if payload:
                logger.error(f"[SERVER] Request payload: {str(payload)[:200]}")
            else:
                logger.error(f"[SERVER] No request payload or invalid JSON")
        except Exception as e:
            logger.error(f"[SERVER] Error logging request details: {str(e)}")
