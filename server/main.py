"""LMS Platform Main Application Module.

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
    allow_origins=[
        "http://localhost:7040",  # local frontend
        "https://lms-swart-five.vercel.app",  # Vercel frontend domain
        "https://lms-30o7ryg5m-gads-projects-02bd6234.vercel.app",  # Vercel backend domain (if needed)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run("server.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 7050)), reload=True)