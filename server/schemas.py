"""Pydantic Schemas Module for LMS Platform.

This module defines data models used for request validation and response
serialization throughout the application. All models extend from Pydantic's
BaseModel to provide automatic validation, serialization, and documentation.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional

class RegisterRequest(BaseModel):
    """Schema for user registration request.
    
    Validates incoming registration requests, ensuring they contain
    the required username and password fields.
    
    Attributes:
        username: Unique identifier for the user
        password: User's password (will be hashed before storage)
        role: Optional role for the user
        phone_number: Optional phone number for the user
    """
    username: str
    password: str
    role: Optional[int] = None
    phone_number: Optional[int] = Field(default=None, description="User's phone number")

class UserResponse(BaseModel):
    """Schema for returning user information.
    
    Excludes sensitive data like passwords.
    
    Attributes:
        userId: Unique identifier for the user
        username: User's username
        max_file_size: Maximum allowed file size in bytes for the user
        role: User's role
        phone_number: Optional phone number for the user
    """
    userId: int
    username: str
    max_file_size: int
    role: int
    phone_number: Optional[int] = None

    model_config = ConfigDict(from_attributes=True) #  Ensures compatibility with SQLAlchemy models
