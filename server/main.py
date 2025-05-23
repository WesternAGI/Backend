from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from server.routes import router

app = FastAPI(
    docs_url="/api-docs",
    redoc_url="/api-redoc"
)

# List of allowed origins - add your frontend URL here
# For development:
ALLOWED_ORIGINS = [
    "http://localhost:3000"
    
]

# Use only the built-in CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # Expose all headers to the client
)

# Remove the custom CORS middleware as it's not needed and can cause issues

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run("server.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 7050)), reload=True)