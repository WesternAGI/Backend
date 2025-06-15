"""
Utility functions and helpers for the backend application.

This package contains various utility modules:
- logging_utils.py: Logging configuration and helper functions
- Other utility modules can be added here
"""

# Make logging utilities available at the package level
from .logging_utils import (
    logger,
    log_request_start,
    log_request_payload,
    log_validation,
    log_error,
    log_response,
    log_ai_call,
    log_ai_response,
    log_something
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
