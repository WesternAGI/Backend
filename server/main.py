from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from server.routes import router

app = FastAPI(
    docs_url="/api-docs",
    redoc_url="/api-redoc"
)

# List of allowed origins
ALLOWED_ORIGINS = [
    "http://localhost:3000",  # Local development
    "https://thoth-frontend-sable.vercel.app", 
    "https://web-production-d7d37.up.railway.app"  # Your backend domain
]

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https?://(localhost(:\d+)?|thothfrontend\.vercel\.app|.*-thoth\.vercel\.app|.*\.vercel\.app|web-production-d7d37\.up\.railway\.app)",  # Allow Vercel and localhost
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "Access-Control-Allow-Headers",
        "Access-Control-Allow-Origin",
        "X-Requested-With",
        "X-CSRF-Token",
        "Accept",
        "Accept-Version",
        "Content-Length",
        "Content-MD5",
        "Date",
        "X-Api-Version",
        "X-Request-Id",
        "X-Forwarded-For",
        "X-Forwarded-Proto",
        "X-Forwarded-Host",
        "X-Forwarded-Port",
        "X-Forwarded-Prefix",
    ],
    expose_headers=[
        "Content-Range",
        "X-Total-Count",
        "Link",
        "X-Request-Id",
        "X-Response-Time",
    ],
    max_age=600,  # 10 minutes
)

# Add a simple health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Include your router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7050)),
        reload=True,
        timeout_keep_alive=300,  # Increase keep-alive timeout
        proxy_headers=True,  # Trust X-Forwarded-* headers
    )