from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.retriever import get_retriever
from src.config import config
from src.market_data import get_market_summary
from typing import Optional, Generator

llm = ChatOllama(model=config.model_name, temperature=0, base_url=config.ollama_host)

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
- NEVER expose data belonging to other users. Only discuss information relevant to the current user.

Answer the user's question using the context and live data below.
If neither contains enough information, say: "I don't have enough information about that. You can visit {config.public_url} for more details."

Context:
{{context}}

Live Data:
{{live_data}}"""

MARKET_SYSTEM = f"""You are a market intelligence AI assistant for FoodBundles restaurant owners.
You have access to real-time commodity price data from Rwanda's markets (RBA data).
Help restaurant owners make smart purchasing decisions based on current prices, trends, and peer benchmarks.

RULES:
- Only discuss market data, commodity prices, buying advice, and cost optimisation.
- NEVER reveal other restaurants' identities or specific financial data.
- NEVER expose data belonging to other users. Only discuss information relevant to the current authenticated user.
- Keep responses actionable and concise.
- Always refer to {config.public_url} for placing orders.

Authenticated user: {{user_context}}

Current Market Data:
{{market_data}}

Conversation context:
{{context}}"""

SUPPORT_SYSTEM = f"""You are a customer support AI for FoodBundles restaurant owners.
You help with questions about orders, vouchers, wallet top-ups, subscriptions, and account management.

RULES:
- Only discuss information relevant to the current authenticated user (identified below).
- NEVER reveal other users' data, orders, or financial information.
- For account-specific actions (cancel order, top up wallet), direct users to the app at {config.public_url}.
- Keep responses helpful, empathetic, and concise.

Authenticated user: {{user_context}}

Platform context:
{{context}}"""

ASSISTANCE_SYSTEM = f"""You are an AI assistant embedded in the FoodBundles restaurant app.
You help the authenticated restaurant owner with in-app tasks: understanding their orders, market prices, wallet, vouchers, and subscriptions.

RULES:
- Only discuss information relevant to the current authenticated user (identified below).
- NEVER reveal other users' data or internal system details.
- Keep responses concise and actionable.
- Always refer to {config.public_url} for actions outside the app.

Authenticated user: {{user_context}}

App context provided by the screen:
{{context}}"""


def _build_history(history: list) -> list:
    messages = []
    for msg in history[:-1]:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages


def is_foodbundles_query(query: str, history: list) -> bool:
    if any(kw in query.lower() for kw in FOODBUNDLES_KEYWORDS):
        return True
    recent = " ".join(m.get("content", "") for m in history[-4:]).lower()
    return any(kw in recent for kw in FOODBUNDLES_KEYWORDS)


def is_sensitive_query(query: str) -> bool:
    return any(p in query.lower() for p in SENSITIVE_PATTERNS)


def _get_rag_context(query: str) -> str:
    try:
        retriever = get_retriever()
        docs = retriever.invoke(query)
        safe_docs = [d for d in docs if not d.metadata.get("internal", False)]
        return "\n\n".join(d.page_content for d in safe_docs)
    except Exception:
        return ""


def run_agent(
    query: str,
    history: list = None,
    user_id: Optional[str] = None,
    role: Optional[str] = None,
) -> str:
    history = history or []
    if is_sensitive_query(query):
        return f"That information is not available. For everything you need, please visit [{config.public_url}]({config.public_url})"

    history_messages = _build_history(history)

    if is_foodbundles_query(query, history):
        context = _get_rag_context(query)
        messages = (
            [SystemMessage(content=RAG_SYSTEM.format(context=context, live_data="No live data available."))]
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


def _user_context_str(user_id: Optional[str], role: Optional[str]) -> str:
    if user_id:
        return f"User ID: {user_id}, Role: {role or 'RESTAURANT'}"
    return "Unauthenticated user"


def run_market_agent(
    query: str,
    history: list = None,
    user_id: Optional[str] = None,
    role: Optional[str] = None,
) -> str:
    history = history or []
    if is_sensitive_query(query):
        return "That information is not available."

    history_messages = _build_history(history)
    market_data = get_market_summary()
    rag_context = _get_rag_context(query)

    system = MARKET_SYSTEM.format(
        user_context=_user_context_str(user_id, role),
        market_data=market_data,
        context=rag_context,
    )
    messages = (
        [SystemMessage(content=system)]
        + history_messages
        + [HumanMessage(content=query)]
    )
    return llm.invoke(messages).content


def run_support_agent(
    query: str,
    history: list = None,
    user_id: Optional[str] = None,
    role: Optional[str] = None,
) -> str:
    history = history or []
    if is_sensitive_query(query):
        return "That information is not available."

    history_messages = _build_history(history)
    rag_context = _get_rag_context(query)
    context = f"Platform knowledge:\n{rag_context}"

    system = SUPPORT_SYSTEM.format(
        user_context=_user_context_str(user_id, role),
        context=context,
    )
    messages = (
        [SystemMessage(content=system)]
        + history_messages
        + [HumanMessage(content=query)]
    )
    return llm.invoke(messages).content


def run_assistance_agent(
    query: str,
    history: list = None,
    user_id: Optional[str] = None,
    role: Optional[str] = None,
    context_data: Optional[dict] = None,
) -> str:
    history = history or []
    if is_sensitive_query(query):
        return "That information is not available."

    history_messages = _build_history(history)
    context_str = ", ".join(f"{k}: {v}" for k, v in context_data.items()) if context_data else ""

    system = ASSISTANCE_SYSTEM.format(
        user_context=_user_context_str(user_id, role),
        context=context_str or "No additional context provided.",
    )
    messages = (
        [SystemMessage(content=system)]
        + history_messages
        + [HumanMessage(content=query)]
    )
    return llm.invoke(messages).content


# --- Streaming variants (yield tokens) ---

def stream_agent(
    query: str,
    history: list = None,
    user_id: Optional[str] = None,
    role: Optional[str] = None,
) -> Generator[str, None, None]:
    history = history or []
    if is_sensitive_query(query):
        yield f"That information is not available. Visit [{config.public_url}]({config.public_url})"
        return

    history_messages = _build_history(history)
    if is_foodbundles_query(query, history):
        context = _get_rag_context(query)
        messages = (
            [SystemMessage(content=RAG_SYSTEM.format(context=context, live_data="No live data available."))]
            + history_messages
            + [HumanMessage(content=query)]
        )
    else:
        messages = (
            [SystemMessage(content=GENERAL_SYSTEM)]
            + history_messages
            + [HumanMessage(content=query)]
        )
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


def stream_market_agent(
    query: str,
    history: list = None,
    user_id: Optional[str] = None,
    role: Optional[str] = None,
) -> Generator[str, None, None]:
    history = history or []
    if is_sensitive_query(query):
        yield "That information is not available."
        return

    history_messages = _build_history(history)
    market_data = get_market_summary()
    rag_context = _get_rag_context(query)
    system = MARKET_SYSTEM.format(
        user_context=_user_context_str(user_id, role),
        market_data=market_data,
        context=rag_context,
    )
    messages = (
        [SystemMessage(content=system)]
        + history_messages
        + [HumanMessage(content=query)]
    )
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


def stream_support_agent(
    query: str,
    history: list = None,
    user_id: Optional[str] = None,
    role: Optional[str] = None,
) -> Generator[str, None, None]:
    history = history or []
    if is_sensitive_query(query):
        yield "That information is not available."
        return

    history_messages = _build_history(history)
    rag_context = _get_rag_context(query)
    context = f"Platform knowledge:\n{rag_context}"
    system = SUPPORT_SYSTEM.format(
        user_context=_user_context_str(user_id, role),
        context=context,
    )
    messages = (
        [SystemMessage(content=system)]
        + history_messages
        + [HumanMessage(content=query)]
    )
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


def stream_assistance_agent(
    query: str,
    history: list = None,
    user_id: Optional[str] = None,
    role: Optional[str] = None,
    context_data: Optional[dict] = None,
) -> Generator[str, None, None]:
    history = history or []
    if is_sensitive_query(query):
        yield "That information is not available."
        return

    history_messages = _build_history(history)
    context_str = ", ".join(f"{k}: {v}" for k, v in context_data.items()) if context_data else ""
    system = ASSISTANCE_SYSTEM.format(
        user_context=_user_context_str(user_id, role),
        context=context_str or "No additional context provided.",
    )
    messages = (
        [SystemMessage(content=system)]
        + history_messages
        + [HumanMessage(content=query)]
    )
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content
