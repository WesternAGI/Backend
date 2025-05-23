"""Thoth Platform Main Application Module.

This module initializes the FastAPI application, sets up CORS middleware,
and includes routes. It serves as the entry point for the LMS platform.
"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from server.routes import router

app = FastAPI(
    docs_url="/api-docs",  # Change FastAPI automatic docs URL to avoid conflict with MkDocs
    redoc_url="/api-redoc"
)

# Keep the CORS middleware but with allow_origins=[]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],  # Will be handled by our custom middleware
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware to handle CORS
@app.middleware("http")
async def cors_middleware(request: Request, call_next):
    # Handle preflight requests
    if request.method == "OPTIONS":
        response = JSONResponse(
            status_code=200,
            content={"status": "ok"}
        )
    else:
        response = await call_next(request)
    
    # Get the origin from the request
    origin = request.headers.get("origin")
    
    # Set CORS headers
    response.headers["Access-Control-Allow-Origin"] = origin or "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    
    return response

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run("server.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 7050)), reload=True)