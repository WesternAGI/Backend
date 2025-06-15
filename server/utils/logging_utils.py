"""Logging utilities for the backend application."""

import logging
from typing import Optional, Dict, Any

# Set up logger
logger = logging.getLogger("lms.server")
logger.setLevel(logging.INFO)

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)


def log_request_start(endpoint: str, method: str, headers: Dict[str, str], client_host: str) -> None:
    """Log the start of an incoming request.
    
    Args:
        endpoint: The API endpoint being called
        method: HTTP method (GET, POST, etc.)
        headers: Request headers
        client_host: Client IP address
    """
    logger.info(
        "Request started: %s %s from %s\nHeaders: %s",
        method, endpoint, client_host, headers
    )


def log_request_payload(payload: Any, endpoint: str) -> None:
    """Log the payload of a request.
    
    Args:
        payload: The request payload (usually a dict)
        endpoint: The API endpoint being called
    """
    logger.debug("Request payload for %s: %s", endpoint, payload)


def log_validation(field: str, value: Any, valid: bool, endpoint: str) -> None:
    """Log field validation results.
    
    Args:
        field: Name of the field being validated
        value: The field value
        valid: Whether validation passed
        endpoint: The API endpoint being called
    """
    if not valid:
        logger.warning(
            "Validation failed for field '%s' with value '%s' in endpoint %s",
            field, value, endpoint
        )


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
