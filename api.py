from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from collections import defaultdict
from typing import Optional, AsyncGenerator
from src.agents import run_agent, run_market_agent, run_support_agent, run_assistance_agent, is_sensitive_query, stream_agent, stream_market_agent, stream_support_agent, stream_assistance_agent
from src.config import config
import httpx
import time
import json

app = FastAPI(
    title="FoodBundles AI Assistant API",
    description="AI assistant API for the FoodBundles platform.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Transfer-Encoding"],
)

# Rate limiting: 60 req/min per IP
RATE_LIMIT = 60
RATE_WINDOW = 60
_request_counts: dict = defaultdict(list)


def _get_client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def check_rate_limit(ip: str):
    now = time.time()
    _request_counts[ip] = [t for t in _request_counts[ip] if now - t < RATE_WINDOW]
    if len(_request_counts[ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests.")
    _request_counts[ip].append(now)


# --- Models ---
class QueryRequest(BaseModel):
    question: str
    history: list[dict] = []


class AuthenticatedQueryRequest(BaseModel):
    question: str
    history: list[dict] = []
    user_id: Optional[str] = None
    role: Optional[str] = None
    context: Optional[dict] = None
    db_context: Optional[dict] = None  # live DB data injected by the backend


class QueryResponse(BaseModel):
    answer: str
    platform_url: str = config.public_url


# --- Health ---
@app.get("/", tags=["Health"])
def root():
    return {"service": "FoodBundles AI Assistant", "status": "running", "platform": config.public_url}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


@app.get("/health/ollama", tags=["Health"])
def health_ollama():
    """Check Ollama connectivity and list available models."""
    try:
        resp = httpx.get(f"{config.ollama_host}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return {"status": "ok", "ollama_host": config.ollama_host, "model": config.model_name, "embedding_model": config.embedding_model, "available_models": models}
    except Exception as e:
        return {"status": "error", "ollama_host": config.ollama_host, "model": config.model_name, "error": str(e)}


@app.get("/health/rag", tags=["Health"])
def health_rag():
    """Check whether the vectorstore is available and RAG retrieval works."""
    import os
    vectorstore_exists = os.path.isdir(config.vectorstore_path)
    if not vectorstore_exists:
        return {"status": "no_vectorstore", "path": config.vectorstore_path, "hint": "Run ingestion first: python -m src.ingestion"}
    try:
        from src.retriever import get_retriever
        retriever = get_retriever()
        docs = retriever.invoke("FoodBundles")
        return {"status": "ok", "vectorstore_path": config.vectorstore_path, "sample_chunks": len(docs)}
    except Exception as e:
        return {"status": "error", "vectorstore_path": config.vectorstore_path, "error": str(e)}


# --- Legacy public endpoints ---
@app.get("/query", tags=["Chat"])
def query_get(q: str, request: Request):
    check_rate_limit(_get_client_ip(request))
    if is_sensitive_query(q):
        return JSONResponse(status_code=403, content={"error": "Restricted content.", "platform_url": config.public_url})
    return QueryResponse(answer=run_agent(q))


@app.post("/query", tags=["Chat"])
def query_post(body: QueryRequest, request: Request):
    check_rate_limit(_get_client_ip(request))
    if is_sensitive_query(body.question):
        return JSONResponse(status_code=403, content={"error": "Restricted content.", "platform_url": config.public_url})
    return QueryResponse(answer=run_agent(body.question, body.history))


# --- Mobile Restaurant AI endpoints (non-streaming) ---

@app.post("/mobile/restaurant/chat", tags=["Mobile Restaurant — AI Chat"])
def mobile_chat(body: AuthenticatedQueryRequest, request: Request):
    check_rate_limit(_get_client_ip(request))
    if is_sensitive_query(body.question):
        return JSONResponse(status_code=403, content={"error": "Restricted content.", "platform_url": config.public_url})
    answer = run_agent(body.question, body.history, user_id=body.user_id, role=body.role, db_context=body.db_context)
    return {"answer": answer, "platform_url": config.public_url}


@app.post("/mobile/restaurant/market-intelligence", tags=["Mobile Restaurant — Market Intelligence"])
def mobile_market_intelligence(body: AuthenticatedQueryRequest, request: Request):
    check_rate_limit(_get_client_ip(request))
    if is_sensitive_query(body.question):
        return JSONResponse(status_code=403, content={"error": "Restricted content.", "platform_url": config.public_url})
    answer = run_market_agent(body.question, body.history, user_id=body.user_id, role=body.role)
    return {"answer": answer, "platform_url": config.public_url}


@app.post("/mobile/restaurant/support", tags=["Mobile Restaurant — Support Chat"])
def mobile_support(body: AuthenticatedQueryRequest, request: Request):
    check_rate_limit(_get_client_ip(request))
    if is_sensitive_query(body.question):
        return JSONResponse(status_code=403, content={"error": "Restricted content.", "platform_url": config.public_url})
    answer = run_support_agent(body.question, body.history, user_id=body.user_id, role=body.role, db_context=body.db_context)
    return {"answer": answer, "platform_url": config.public_url}


@app.post("/mobile/restaurant/assistance", tags=["Mobile Restaurant — AI Assistance"])
def mobile_assistance(body: AuthenticatedQueryRequest, request: Request):
    check_rate_limit(_get_client_ip(request))
    if is_sensitive_query(body.question):
        return JSONResponse(status_code=403, content={"error": "Restricted content.", "platform_url": config.public_url})
    answer = run_assistance_agent(body.question, body.history, user_id=body.user_id, role=body.role, context_data=body.context, db_context=body.db_context)
    return {"answer": answer, "platform_url": config.public_url}


# --- Streaming SSE endpoints ---

async def _sse_stream(gen) -> AsyncGenerator[str, None]:
    """Wraps a sync token generator into SSE data: chunks."""
    try:
        for token in gen:
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post("/mobile/restaurant/chat/stream", tags=["Mobile Restaurant — AI Chat"])
def mobile_chat_stream(body: AuthenticatedQueryRequest, request: Request):
    """SSE streaming version of /mobile/restaurant/chat."""
    check_rate_limit(_get_client_ip(request))
    if is_sensitive_query(body.question):
        return JSONResponse(status_code=403, content={"error": "Restricted content."})
    gen = stream_agent(body.question, body.history, user_id=body.user_id, role=body.role, db_context=body.db_context)
    return StreamingResponse(_sse_stream(gen), media_type="text/event-stream")


@app.post("/mobile/restaurant/market-intelligence/stream", tags=["Mobile Restaurant — Market Intelligence"])
def mobile_market_intelligence_stream(body: AuthenticatedQueryRequest, request: Request):
    """SSE streaming version of /mobile/restaurant/market-intelligence."""
    check_rate_limit(_get_client_ip(request))
    if is_sensitive_query(body.question):
        return JSONResponse(status_code=403, content={"error": "Restricted content."})
    gen = stream_market_agent(body.question, body.history, user_id=body.user_id, role=body.role)
    return StreamingResponse(_sse_stream(gen), media_type="text/event-stream")


@app.post("/mobile/restaurant/support/stream", tags=["Mobile Restaurant — Support Chat"])
def mobile_support_stream(body: AuthenticatedQueryRequest, request: Request):
    """SSE streaming version of /mobile/restaurant/support."""
    check_rate_limit(_get_client_ip(request))
    if is_sensitive_query(body.question):
        return JSONResponse(status_code=403, content={"error": "Restricted content."})
    gen = stream_support_agent(body.question, body.history, user_id=body.user_id, role=body.role, db_context=body.db_context)
    return StreamingResponse(_sse_stream(gen), media_type="text/event-stream")


@app.post("/mobile/restaurant/assistance/stream", tags=["Mobile Restaurant — AI Assistance"])
def mobile_assistance_stream(body: AuthenticatedQueryRequest, request: Request):
    """SSE streaming version of /mobile/restaurant/assistance."""
    check_rate_limit(_get_client_ip(request))
    if is_sensitive_query(body.question):
        return JSONResponse(status_code=403, content={"error": "Restricted content."})
    gen = stream_assistance_agent(body.question, body.history, user_id=body.user_id, role=body.role, context_data=body.context, db_context=body.db_context)
    return StreamingResponse(_sse_stream(gen), media_type="text/event-stream")
