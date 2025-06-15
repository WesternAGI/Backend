"""
Utility functions and helpers for the backend application.

This package contains various utility modules:
- logging_utils.py: Logging configuration and helper functions
- Other utility modules can be added here
"""

# Make logging utilities available at the package level
from .logging_utils import (
    logger,
    log_server_lifecycle,
    log_server_health,
    log_request_start,
    log_request_payload,
    log_validation,
    log_error,
    log_response,
    log_ai_call,
    log_ai_response,
    log_something,
    log_file_operation
)

# Export the main utility functions
__all__ = [
    'logger',
    'log_request_start',
    'log_request_payload',
    'log_validation',
    'log_error',
    'log_response',
    'log_ai_call',
    'log_ai_response',
    'log_something',
]
