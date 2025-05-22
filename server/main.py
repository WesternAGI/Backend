"""Thoth Platform Main Application Module.

This module initializes the FastAPI application, sets up CORS middleware,
and includes routes. It serves as the entry point for the LMS platform.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.routes import router

app = FastAPI(
    docs_url="/api-docs",  # Change FastAPI automatic docs URL to avoid conflict with MkDocs
    redoc_url="/api-redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run("server.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 7050)), reload=True)