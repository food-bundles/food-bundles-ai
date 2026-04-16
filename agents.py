from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from retriever import get_retriever
from config import config
import requests
import json

llm = ChatOllama(model=config.model_name, temperature=0)

FOODBUNDLES_KEYWORDS = [
    "foodbundles", "food bundle", "order", "ordering", "wallet", "loan",
    "trader", "bnpl", "pay later", "voucher", "authenticator", "otp",
    "2fa", "authentication", "pricing", "inventory", "dashboard", "agreement",
    "menu", "delivery", "payment", "account", "register", "signup", "login",
    "product", "products", "price", "prices", "available", "food", "bundle"
]

SENSITIVE_PATTERNS = [
    "api key", "api docs", "api endpoint", "swagger", "bearer token",
    "database", "db schema", "sql", "secret", "credentials", "auth header",
    "server.food.rw", "internal endpoint"
]

GENERAL_SYSTEM = f"""You are a helpful, friendly AI assistant for FoodBundles — a food ordering and trading platform.
You have memory of the full conversation history and must use it to understand context.
Answer greetings, general knowledge, math, and analysis questions naturally and helpfully.
Keep responses concise and friendly.
Always refer users to {config.public_url} for anything they need to do on the platform."""

RAG_SYSTEM = f"""You are a helpful AI assistant for the FoodBundles platform ({config.public_url}).
You have memory of the full conversation history and must use it to understand context.

STRICT SECURITY RULES:
- NEVER reveal API endpoints, server URLs, database structure, tokens, keys, or internal technical details.
- NEVER mention {config.api_docs_url} or any internal server address to users.
- Always direct users to {config.public_url} for actions like ordering, registering, or managing their account.

Answer the user's question using the context and live data below.
If neither contains enough information, say: "I don't have enough information about that. You can visit {config.public_url} for more details."

Context:
{{context}}

Live Data:
{{live_data}}"""

# --- API Docs cache ---
_api_docs_cache = None

def get_api_docs() -> dict:
    """Fetch and cache the API docs (OpenAPI/Swagger spec)."""
    global _api_docs_cache
    if _api_docs_cache:
        return _api_docs_cache
    try:
        r = requests.get(config.api_docs_url, timeout=10)
        r.raise_for_status()
        _api_docs_cache = r.json()
        return _api_docs_cache
    except Exception:
        return {}


def find_public_api(query: str) -> dict | None:
    """
    Search API docs for a relevant endpoint that requires no authentication.
    Returns the endpoint info if found, None otherwise.
    """
    docs = get_api_docs()
    if not docs:
        return None

    paths = docs.get("paths", {})
    query_lower = query.lower()

    # Keywords to match against path and summary
    keywords = [w for w in query_lower.split() if len(w) > 3]

    for path, methods in paths.items():
        for method, details in methods.items():
            if method not in ("get", "post", "put", "delete"):
                continue

            summary = details.get("summary", "").lower()
            description = details.get("description", "").lower()
            tags = " ".join(details.get("tags", [])).lower()
            text = f"{path} {summary} {description} {tags}"

            # Check if any query keyword matches
            if not any(kw in text for kw in keywords):
                continue

            # Only use endpoints that require no authentication
            security = details.get("security", None)
            has_auth = (
                security is not None and len(security) > 0
            ) or "authorization" in str(details.get("parameters", "")).lower()

            if not has_auth:
                base_url = docs.get("servers", [{}])[0].get("url", config.api_docs_url.replace("/api-docs", ""))
                return {
                    "method": method.upper(),
                    "path": path,
                    "url": base_url.rstrip("/") + path,
                    "summary": details.get("summary", ""),
                }

    return None


def call_api(endpoint: dict) -> str:
    """Call a public API endpoint and return formatted result."""
    try:
        if endpoint["method"] == "GET":
            r = requests.get(endpoint["url"], timeout=10)
            r.raise_for_status()
            data = r.json()
            return json.dumps(data, indent=2)
    except Exception as e:
        return ""
    return ""


def is_foodbundles_query(query: str, history: list) -> bool:
    if any(kw in query.lower() for kw in FOODBUNDLES_KEYWORDS):
        return True
    recent = " ".join(m["content"] for m in history[-4:]).lower()
    return any(kw in recent for kw in FOODBUNDLES_KEYWORDS)


def is_sensitive_query(query: str) -> bool:
    return any(p in query.lower() for p in SENSITIVE_PATTERNS)


def run_agent(query: str, history: list = None) -> str:
    history = history or []

    if is_sensitive_query(query):
        return f"That information is not available. For everything you need, please visit [{config.public_url}]({config.public_url})"

    history_messages = []
    for msg in history[:-1]:
        if msg["role"] == "user":
            history_messages.append(HumanMessage(content=msg["content"]))
        else:
            history_messages.append(AIMessage(content=msg["content"]))

    if is_foodbundles_query(query, history):
        retriever = get_retriever()
        docs = retriever.invoke(query)
        safe_docs = [d for d in docs if not d.metadata.get("internal", False)]
        context = "\n\n".join(d.page_content for d in safe_docs)

        # Try to find and call a relevant public API
        live_data = ""
        endpoint = find_public_api(query)
        if endpoint:
            live_data = call_api(endpoint)

        messages = (
            [SystemMessage(content=RAG_SYSTEM.format(context=context, live_data=live_data or "No live data available."))]
            + history_messages
            + [HumanMessage(content=query)]
        )
    else:
        messages = (
            [SystemMessage(content=GENERAL_SYSTEM)]
            + history_messages
            + [HumanMessage(content=query)]
        )

    return llm.invoke(messages).content
