from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from config import config


def get_retriever():
    embeddings = OllamaEmbeddings(model=config.embedding_model)
    db = FAISS.load_local(config.vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 5})