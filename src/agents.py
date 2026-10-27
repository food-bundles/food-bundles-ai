"""
AI agents using Ollama (Meta Llama) via direct HTTP API.
Supports tool calling: AI can call backend APIs to fetch live user data on demand.
"""
import json
import os
import httpx
from src.config import config
from src.market_data import get_market_summary
from typing import Optional, Generator

AI_SERVICE_SECRET = os.getenv("AI_SERVICE_SECRET", "")

# ── Helpers ───────────────────────────────────────────────────────────────────

FOODBUNDLES_KEYWORDS = [
    "foodbundles", "food bundle", "order", "ordering", "wallet", "loan",
    "trader", "bnpl", "pay later", "voucher", "authenticator", "otp",
    "2fa", "authentication", "pricing", "inventory", "dashboard", "agreement",
    "menu", "delivery", "payment", "account", "register", "signup", "login",
    "product", "products", "price", "prices", "available", "food", "bundle",
    "market", "commodity", "potato", "tomato", "onion", "cabbage", "carrot",
    "rsi", "momentum", "ohlc", "buying advice", "peer", "ranking", "benchmark",
]

SENSITIVE_PATTERNS = [
    "api key", "api docs", "api endpoint", "swagger", "bearer token",
    "database", "db schema", "sql", "secret", "credentials", "auth header",
    "server.food.rw", "internal endpoint",
]

GENERAL_SYSTEM = (
    f"You are a helpful, friendly AI assistant for FoodBundles — a food ordering and trading platform in Rwanda. "
    f"You have memory of the full conversation history and must use it to understand context. "
    f"Answer greetings, general knowledge, math, and analysis questions naturally and helpfully. "
    f"Keep responses concise and friendly. "
    f"Always refer users to {config.public_url} for anything they need to do on the platform."
)

RAG_SYSTEM_TMPL = (
    f"You are a helpful AI assistant for the FoodBundles platform ({config.public_url}). "
    f"You have memory of the full conversation history and must use it to understand context.\n\n"
    "RULES:\n"
    f"- NEVER reveal API endpoints, server URLs, database structure, tokens, or internal technical details.\n"
    f"- Always direct users to {config.public_url} for actions like ordering, registering, or managing their account.\n"
    "- NEVER expose data belonging to other users.\n\n"
    "Answer the user's question using the context below. "
    f"If it doesn't contain enough information, say: \"I don't have enough information about that. "
    f"You can visit {config.public_url} for more details.\"\n\n"
    "Context:\n{{context}}"
)

MARKET_SYSTEM_TMPL = (
    f"You are a market intelligence AI assistant for FoodBundles restaurant owners in Rwanda. "
    "You have access to real commodity price data from Rwanda's markets (RBA data). "
    "Help restaurant owners make smart purchasing decisions based on current prices, trends, and peer benchmarks.\n\n"
    "RULES:\n"
    "- Only discuss market data, commodity prices, buying advice, and cost optimisation.\n"
    "- NEVER reveal other restaurants' identities or specific financial data.\n"
    "- Keep responses actionable and concise.\n"
    f"- Always refer to {config.public_url} for placing orders.\n\n"
    "Authenticated user: {{user_context}}\n\n"
    "Current Market Data:\n{{market_data}}\n\n"
    "Conversation context:\n{{context}}"
)

SUPPORT_SYSTEM_TMPL = (
    f"You are a customer support AI for FoodBundles restaurant owners in Rwanda. "
    "You help with questions about orders, vouchers, wallet top-ups, subscriptions, and account management.\n\n"
    "RULES:\n"
    "- Only discuss information relevant to the current authenticated user (identified below).\n"
    "- NEVER reveal other users' data, orders, or financial information.\n"
    f"- For account-specific actions (cancel order, top up wallet), direct users to the app at {config.public_url}.\n"
    "- Keep responses helpful, empathetic, and concise.\n\n"
    "Authenticated user: {{user_context}}\n\n"
    "Platform context:\n{{context}}"
)

ASSISTANCE_SYSTEM_TMPL = (
    f"You are an AI assistant embedded in the FoodBundles restaurant app. "
    "You help the authenticated restaurant owner with in-app tasks: understanding their orders, market prices, wallet, vouchers, and subscriptions.\n\n"
    "RULES:\n"
    "- Only discuss information relevant to the current authenticated user (identified below).\n"
    "- NEVER reveal other users' data or internal system details.\n"
    "- Keep responses concise and actionable.\n"
    f"- Always refer to {config.public_url} for actions outside the app.\n\n"
    "Authenticated user: {{user_context}}\n\n"
    "App context provided by the screen:\n{{context}}"
)


def is_foodbundles_query(query: str, history: list) -> bool:
    if any(kw in query.lower() for kw in FOODBUNDLES_KEYWORDS):
        return True
    recent = " ".join(m.get("content", "") for m in history[-4:]).lower()
    return any(kw in recent for kw in FOODBUNDLES_KEYWORDS)


def is_sensitive_query(query: str) -> bool:
    return any(p in query.lower() for p in SENSITIVE_PATTERNS)


def _get_rag_context(query: str) -> str:
    try:
        from src.retriever import get_retriever
        retriever = get_retriever()
        docs = retriever.invoke(query)
        safe_docs = [d for d in docs if not d.metadata.get("internal", False)]
        return "\n\n".join(d.page_content for d in safe_docs)
    except Exception:
        return ""


def _user_context_str(user_id: Optional[str], role: Optional[str]) -> str:
    if user_id:
        return f"User ID: {user_id}, Role: {role or 'RESTAURANT'}"
    return "Unauthenticated user"


def _db_context_str(db_context: Optional[dict]) -> str:
    """Convert the live DB snapshot into readable text for the system prompt."""
    if not db_context:
        return ""
    parts = []
    orders = db_context.get("recent_orders", [])
    if orders:
        parts.append("Recent orders:")
        for o in orders:
            items_str = ", ".join(o.get("items", [])) or "no items"
            parts.append(
                f"  • #{o.get('order_number', o.get('id', '?'))} — {o.get('status')} / {o.get('payment_status')} "
                f"— {o.get('total_rwf')} RWF — placed {o.get('placed_at', '')} — items: {items_str}"
            )
    wallet = db_context.get("wallet")
    if wallet:
        parts.append(f"Wallet balance: {wallet.get('balance_rwf')} {wallet.get('currency', 'RWF')}")
    sub = db_context.get("subscription")
    if sub:
        parts.append(f"Subscription: {sub.get('plan')} ({sub.get('status')}) — valid until {sub.get('valid_until')}")
    card = db_context.get("voucher_card")
    if card:
        parts.append(f"Voucher card: {card.get('status')} — loan limit {card.get('loan_limit_rwf')} RWF")
    return "\n".join(parts) if parts else ""


def _build_ollama_messages(system: str, history: list, query: str) -> list[dict]:
    messages = [{"role": "system", "content": system}]
    for msg in history[:-1]:
        role = msg.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        messages.append({"role": role, "content": msg.get("content", "")})
    messages.append({"role": "user", "content": query})
    return messages


# ── Ollama HTTP API calls ─────────────────────────────────────────────────────

def _ollama_call(system: str, history: list, query: str) -> str:
    """Non-streaming call to Ollama /api/chat endpoint."""
    messages = _build_ollama_messages(system, history, query)
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{config.ollama_host}/api/chat",
                json={
                    "model": config.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.7},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
    except Exception as e:
        raise RuntimeError(f"Ollama call failed: {e}") from e


def _ollama_stream(system: str, history: list, query: str) -> Generator[str, None, None]:
    """Streaming call to Ollama /api/chat endpoint — yields text tokens."""
    messages = _build_ollama_messages(system, history, query)
    with httpx.Client(timeout=120.0) as client:
        with client.stream(
            "POST",
            f"{config.ollama_host}/api/chat",
            json={
                "model": config.model_name,
                "messages": messages,
                "stream": True,
                "options": {"temperature": 0.7},
            },
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break


# ── Tool calling ─────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_recent_orders",
            "description": "Fetch the restaurant's most recent orders from the database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of orders to return (max 20, default 5).",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_wallet",
            "description": "Get the restaurant's current wallet balance.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_subscription",
            "description": "Get the restaurant's active subscription plan details.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_voucher_card",
            "description": "Get the restaurant's voucher card / BNPL loan details.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_detail",
            "description": "Fetch full details of a specific order by ID or order number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order UUID."},
                    "order_number": {"type": "string", "description": "The human-readable order number."},
                },
            },
        },
    },
]


def _execute_tool(tool_name: str, args: dict, restaurant_id: str) -> str:
    """Call the backend tool-call endpoint and return the result as JSON string."""
    if not config.backend_url or not AI_SERVICE_SECRET:
        return json.dumps({"error": "Tool calling not configured"})
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                f"{config.backend_url}/mobile/restaurant/ai/tool-call",
                json={"tool": tool_name, "args": args, "restaurant_id": restaurant_id},
                headers={"x-ai-service-secret": AI_SERVICE_SECRET},
            )
            resp.raise_for_status()
            data = resp.json()
            return json.dumps(data.get("result"))
    except Exception as e:
        return json.dumps({"error": str(e)})


def _ollama_call_with_tools(
    system: str,
    history: list,
    query: str,
    restaurant_id: Optional[str],
    max_tool_rounds: int = 3,
) -> str:
    """Ollama chat with tool calling loop. Falls back to plain call if tool calling unsupported."""
    if not restaurant_id:
        return _ollama_call(system, history, query)

    messages = _build_ollama_messages(system, history, query)

    for _ in range(max_tool_rounds + 1):
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{config.ollama_host}/api/chat",
                    json={
                        "model": config.model_name,
                        "messages": messages,
                        "tools": TOOLS,
                        "stream": False,
                        "options": {"temperature": 0.7},
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Ollama call failed: {e}") from e

        msg = data.get("message", {})
        tool_calls = msg.get("tool_calls", [])

        if not tool_calls:
            return msg.get("content", "")

        # Append assistant message with tool_calls
        messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": tool_calls})

        # Execute each tool and append results
        for tc in tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            result = _execute_tool(tool_name, args, restaurant_id)
            messages.append({"role": "tool", "content": result})

    # Max rounds exceeded — return whatever the last message content was
    return msg.get("content", "")


# ── Non-streaming agents ──────────────────────────────────────────────────────

def run_agent(
    query: str,
    history: Optional[list] = None,
    user_id: Optional[str] = None,
    role: Optional[str] = None,
    db_context: Optional[dict] = None,
) -> str:
    history = history or []
    if is_sensitive_query(query):
        return f"That information is not available. For everything you need, please visit {config.public_url}"
    if is_foodbundles_query(query, history):
        rag = _get_rag_context(query)
        db = _db_context_str(db_context)
        combined = "\n\n".join(filter(None, [rag, db])) or "No additional context available."
        system = RAG_SYSTEM_TMPL.format(context=combined)
        return _ollama_call_with_tools(system, history, query, user_id)
    else:
        system = GENERAL_SYSTEM
    return _ollama_call(system, history, query)


def run_market_agent(
    query: str,
    history: Optional[list] = None,
    user_id: Optional[str] = None,
    role: Optional[str] = None,
) -> str:
    history = history or []
    if is_sensitive_query(query):
        return "That information is not available."
    market_data = get_market_summary()
    rag_context = _get_rag_context(query)
    system = MARKET_SYSTEM_TMPL.format(
        user_context=_user_context_str(user_id, role),
        market_data=market_data,
        context=rag_context or "No additional context.",
    )
    return _ollama_call(system, history, query)


def run_support_agent(
    query: str,
    history: Optional[list] = None,
    user_id: Optional[str] = None,
    role: Optional[str] = None,
    db_context: Optional[dict] = None,
) -> str:
    history = history or []
    if is_sensitive_query(query):
        return "That information is not available."
    rag = _get_rag_context(query)
    db = _db_context_str(db_context)
    combined = "\n\n".join(filter(None, [rag, db])) or "No additional context."
    system = SUPPORT_SYSTEM_TMPL.format(
        user_context=_user_context_str(user_id, role),
        context=combined,
    )
    return _ollama_call_with_tools(system, history, query, user_id)


def run_assistance_agent(
    query: str,
    history: Optional[list] = None,
    user_id: Optional[str] = None,
    role: Optional[str] = None,
    context_data: Optional[dict] = None,
    db_context: Optional[dict] = None,
) -> str:
    history = history or []
    if is_sensitive_query(query):
        return "That information is not available."
    screen_ctx = ", ".join(f"{k}: {v}" for k, v in context_data.items()) if context_data else ""
    db = _db_context_str(db_context)
    combined = "\n\n".join(filter(None, [screen_ctx, db])) or "No additional context provided."
    system = ASSISTANCE_SYSTEM_TMPL.format(
        user_context=_user_context_str(user_id, role),
        context=combined,
    )
    return _ollama_call_with_tools(system, history, query, user_id)


# ── Streaming agents ──────────────────────────────────────────────────────────

def stream_agent(
    query: str,
    history: Optional[list] = None,
    user_id: Optional[str] = None,
    role: Optional[str] = None,
    db_context: Optional[dict] = None,
) -> Generator[str, None, None]:
    history = history or []
    if is_sensitive_query(query):
        yield f"That information is not available. Visit {config.public_url}"
        return
    if is_foodbundles_query(query, history):
        rag = _get_rag_context(query)
        db = _db_context_str(db_context)
        combined = "\n\n".join(filter(None, [rag, db])) or "No additional context."
        system = RAG_SYSTEM_TMPL.format(context=combined)
    else:
        system = GENERAL_SYSTEM
    yield from _ollama_stream(system, history, query)


def stream_market_agent(
    query: str,
    history: Optional[list] = None,
    user_id: Optional[str] = None,
    role: Optional[str] = None,
) -> Generator[str, None, None]:
    history = history or []
    if is_sensitive_query(query):
        yield "That information is not available."
        return
    market_data = get_market_summary()
    rag_context = _get_rag_context(query)
    system = MARKET_SYSTEM_TMPL.format(
        user_context=_user_context_str(user_id, role),
        market_data=market_data,
        context=rag_context or "No additional context.",
    )
    yield from _ollama_stream(system, history, query)


def stream_support_agent(
    query: str,
    history: Optional[list] = None,
    user_id: Optional[str] = None,
    role: Optional[str] = None,
    db_context: Optional[dict] = None,
) -> Generator[str, None, None]:
    history = history or []
    if is_sensitive_query(query):
        yield "That information is not available."
        return
    rag = _get_rag_context(query)
    db = _db_context_str(db_context)
    combined = "\n\n".join(filter(None, [rag, db])) or "No additional context."
    system = SUPPORT_SYSTEM_TMPL.format(
        user_context=_user_context_str(user_id, role),
        context=combined,
    )
    yield from _ollama_stream(system, history, query)


def stream_assistance_agent(
    query: str,
    history: Optional[list] = None,
    user_id: Optional[str] = None,
    role: Optional[str] = None,
    context_data: Optional[dict] = None,
    db_context: Optional[dict] = None,
) -> Generator[str, None, None]:
    history = history or []
    if is_sensitive_query(query):
        yield "That information is not available."
        return
    screen_ctx = ", ".join(f"{k}: {v}" for k, v in context_data.items()) if context_data else ""
    db = _db_context_str(db_context)
    combined = "\n\n".join(filter(None, [screen_ctx, db])) or "No additional context provided."
    system = ASSISTANCE_SYSTEM_TMPL.format(
        user_context=_user_context_str(user_id, role),
        context=combined,
    )
    yield from _ollama_stream(system, history, query)
