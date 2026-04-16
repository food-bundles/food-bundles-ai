from dataclasses import dataclass

@dataclass
class Config:
    data_path: str = "./data"
    vectorstore_path: str = "./vectorstore"
    model_name: str = "llama3.2"
    embedding_model: str = "nomic-embed-text"
    api_docs_url: str = "https://server.food.rw/api-docs"
    public_url: str = "https://food.rw"

config = Config()
