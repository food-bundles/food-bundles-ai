from dataclasses import dataclass, field
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass
class Config:
    data_path: str = "./data"
    vectorstore_path: str = "./vectorstore"
    # Meta Llama via Ollama (free, local on the server)
    model_name: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2"))
    embedding_model: str = field(default_factory=lambda: os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"))
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    # Backend callback for tool calling (AI → backend to fetch live data)
    backend_url: str = field(default_factory=lambda: os.getenv("BACKEND_URL", "http://localhost:4000"))
    api_docs_url: str = "https://server.food.rw/api-docs"
    public_url: str = "https://food.rw"

config = Config()
