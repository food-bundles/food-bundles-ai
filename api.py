from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from collections import defaultdict
from src.agents import run_agent, is_sensitive_query
from src.config import config
import time

app = FastAPI(
    title="FoodBundles AI Assistant API",
    description="Public AI assistant API for the FoodBundles platform. Sensitive data is never exposed.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- Rate Limiting (60 requests per minute per IP) ---
RATE_LIMIT = 60
RATE_WINDOW = 60  # seconds
_request_counts: dict = defaultdict(list)

def check_rate_limit(ip: str):
    now = time.time()
    timestamps = _request_counts[ip]
    # Remove timestamps outside the window
    _request_counts[ip] = [t for t in timestamps if now - t < RATE_WINDOW]
    if len(_request_counts[ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")
    _request_counts[ip].append(now)


# --- Request/Response Models ---
class QueryRequest(BaseModel):
    question: str
    history: list[dict] = []

class QueryResponse(BaseModel):
    answer: str
    platform_url: str = config.public_url


# --- Routes ---
@app.get("/", tags=["Health"])
def root():
    return {
        "service": "FoodBundles AI Assistant",
        "status": "running",
        "platform": config.public_url,
    }

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}

@app.get("/query", tags=["Chat"])
def query_get(q: str, request: Request):
    """Simple GET query endpoint for quick questions."""
    check_rate_limit(request.client.host)
    if is_sensitive_query(q):
        return JSONResponse(status_code=403, content={
            "error": "This query contains restricted content.",
            "platform_url": config.public_url,
        })
    return QueryResponse(answer=run_agent(q))

@app.post("/query", tags=["Chat"])
def query_post(body: QueryRequest, request: Request):
    """POST query endpoint with conversation history support."""
    check_rate_limit(request.client.host)
    if is_sensitive_query(body.question):
        return JSONResponse(status_code=403, content={
            "error": "This query contains restricted content.",
            "platform_url": config.public_url,
        })
    return QueryResponse(answer=run_agent(body.question, body.history))
