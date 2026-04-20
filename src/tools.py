from langchain.tools import tool
from src.retriever import get_retriever

@tool
def search_documents(query: str) -> str:
    """Search documents for relevant context"""
    retriever = get_retriever()
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])

@tool
def summarize_text(text: str) -> str:
    """Summarize given text"""
    return text[:500] + "..."