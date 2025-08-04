import logging
import sys
import traceback
from time import perf_counter
from fastapi import FastAPI, Request, Response, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import os
import json
from server.routes import router
from server.utils.logging_utils import (
    log_request_start, 
    log_response,
    log_error,
    log_request_payload
)

# Create logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Clear any existing handlers
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
    handler.close()

# Custom formatter that handles missing request_id
class RequestIdFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, 'request_id'):
            record.request_id = 'NO_REQUEST_ID'
        return super().format(record)

# Create formatters
file_formatter = RequestIdFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# File handler for all logs
file_handler = logging.FileHandler(os.path.join(log_dir, 'server.log'))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(file_formatter)

# Console handler for WARNING and above
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_formatter)

# Add handlers to root logger
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Create app logger
app_logger = logging.getLogger('thoth')
app_logger.setLevel(logging.DEBUG)

# Add custom filter to ensure request_id is set
class RequestIdFilter(logging.Filter):
    def __init__(self, request_id='NO_REQUEST_ID'):
        super().__init__()
        self.request_id = request_id
        
    def filter(self, record):
        if not hasattr(record, 'request_id'):
            record.request_id = self.request_id
        return True

# Configure app logger with custom formatter
formatter = RequestIdFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create a default request ID filter
default_filter = RequestIdFilter()

# Configure file handler for the app logger
file_handler = logging.FileHandler(os.path.join(log_dir, 'thoth.log'))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
file_handler.addFilter(default_filter)

# Configure console handler for the app logger
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_formatter)
console_handler.addFilter(default_filter)

# Clear existing handlers and add the new ones
app_logger.handlers = [file_handler, console_handler]

# Set propagate to False to prevent duplicate logs
app_logger.propagate = False

# Log startup information
app_logger.info("=" * 80)
app_logger.info("Starting Thoth Backend Server")
app_logger.info(f"Python version: {sys.version}")
app_logger.info(f"Working directory: {os.getcwd()}")
app_logger.info(f"Log directory: {log_dir}")
app_logger.info("=" * 80)

# Application metadata
APP_TITLE = "AI-Powered Backend API"
APP_DESCRIPTION = """
A FastAPI-based backend service
"""
APP_VERSION = "1.0.0"
API_PREFIX = "/api"

# Initialize FastAPI with custom OpenAPI configuration
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
    openapi_url=f"{API_PREFIX}/openapi.json",
    contact={
        "name": "Gad Gad",
        "email": "ggad@uwo.ca"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    },
    openapi_tags=[
        {
            "name": "auth",
            "description": "Authentication and user management endpoints"
        },
        {
            "name": "files",
            "description": "File upload and management endpoints"
        },
        {
            "name": "ai",
            "description": "AI-powered query endpoints"
        },
        {
            "name": "devices",
            "description": "Device management and tracking"
        },
        {
            "name": "twilio",
            "description": "Twilio webhook endpoints"
        }
    ]
)

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=APP_TITLE,
        version=APP_VERSION,
        description=APP_DESCRIPTION,
        routes=app.routes,
    )
    
    # Add error responses to all endpoints
    for path in openapi_schema["paths"].values():
        for method in path.values():
            if "responses" not in method:
                method["responses"] = {}
            if "400" not in method["responses"]:
                method["responses"]["400"] = {
                    "description": "Bad Request",
                    "content": {"application/json": {"example": {"detail": "Invalid request data"}}}
                }
            if "401" not in method["responses"]:
                method["responses"]["401"] = {
                    "description": "Unauthorized",
                    "content": {"application/json": {"example": {"detail": "Not authenticated"}}}
                }
            if "403" not in method["responses"]:
                method["responses"]["403"] = {
                    "description": "Forbidden",
                    "content": {"application/json": {"example": {"detail": "Not enough permissions"}}}
                }
            if "404" not in method["responses"]:
                method["responses"]["404"] = {
                    "description": "Not Found",
                    "content": {"application/json": {"example": {"detail": "Item not found"}}}
                }
            if "422" not in method["responses"]:
                method["responses"]["422"] = {
                    "description": "Validation Error",
                    "content": {"application/json": {"example": {"detail": [{"loc": ["string", 0], "msg": "string", "type": "string"}]}}}
                }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Custom docs endpoints
@app.get("/api-docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )

@app.get("/api-redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
    )

# List of allowed origins
ALLOWED_ORIGINS = [
    "http://localhost:3000",  # Local development
    "https://thoth-frontend-sable.vercel.app", 
    "https://web-production-d7d37.up.railway.app"  # Your backend domain
]

# --------------------------------------------------
# Global logging middleware
# --------------------------------------------------
from time import perf_counter
from server.utils.logging_utils import (
    log_request_start,
    log_request_payload,
    log_response,
    log_error,
    logger as app_logger,
)

@app.middleware("http")
async def global_logging_middleware(request: Request, call_next):
    """Middleware that logs every request and response with latency."""
    import uuid
    request_id = str(uuid.uuid4())
    start = perf_counter()
    endpoint = request.url.path
    
    # Add request ID to logger context
    logger = logging.LoggerAdapter(app_logger, {'request_id': request_id})
    
    # Skip logging for health checks to reduce noise
    if endpoint == "/health":
        return await call_next(request)

    # Read body (non-stream) for small payloads ONLY (< 10 kB)
    body_str = ""
    try:
        body_bytes = await request.body()
        if body_bytes and len(body_bytes) <= 10_240:  # 10 KB safety limit
            try:
                body_str = body_bytes.decode("utf-8", errors="ignore")
                # Try to pretty print JSON if possible
                try:
                    json_body = json.loads(body_str)
                    body_str = json.dumps(json_body, indent=2)
                except (json.JSONDecodeError, TypeError):
                    pass
            except Exception as e:
                body_str = f"<binary data: {str(e)[:200]}>"
        elif body_bytes:
            body_str = f"<{len(body_bytes)} bytes>"
    except Exception as e:
        body_str = f"<error reading body: {str(e)}>"

    # Log the request with additional context
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get('user-agent', 'unknown')
    
    logger.info(
        "[REQ] %s %s from %s | UA: %s | Body: %s", 
        request.method, 
        endpoint, 
        client_ip,
        user_agent,
        body_str if body_str else "<no body>"
    )

    try:
        response = await call_next(request)
        duration_ms = (perf_counter() - start) * 1000
        
        # Log response summary
        status_code = getattr(response, "status_code", "<no response>")
        logger.info(
            "[RESP] %s %s -> %d | %.2f ms", 
            request.method, 
            endpoint, 
            status_code,
            duration_ms
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        return response
        
    except Exception as exc:
        duration_ms = (perf_counter() - start) * 1000
        logger.error(
            "[ERROR] %s %s failed after %.2f ms: %s\n%s", 
            request.method, 
            endpoint, 
            duration_ms, 
            str(exc),
            traceback.format_exc()
        )
        raise

# --------------------------------------------------
# Enhanced CORS middleware
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"(https?://(localhost(:\d+)?|thothfrontend\.vercel\.app|.*-thoth\.vercel\.app|.*\.vercel\.app|web-production-d7d37\.up\.railway\.app)|chrome-extension://nnhcocdhioccnhcbjflcdnicmjlbcnbd)",  # Allow Vercel, localhost, and Chrome extension
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

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to the AI-Powered Backend API",
        "status": "operational",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi_schema": "/openapi.json"
        },
        "health_check": "/health"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Include API router without prefix
app.include_router(router, prefix="")

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