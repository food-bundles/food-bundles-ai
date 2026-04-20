from dataclasses import dataclass, field
import os

@dataclass
class Config:
    data_path: str = "./data"
    vectorstore_path: str = "./vectorstore"
    model_name: str = "llama3.2"
    embedding_model: str = "nomic-embed-text"
    api_docs_url: str = "https://server.food.rw/api-docs"
    public_url: str = "https://food.rw"
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))

config = Config()
