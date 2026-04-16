from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import config
import requests


def fetch_api_docs() -> list:
    """Fetch API documentation from the server. Used internally only — never exposed to users."""
    try:
        response = requests.get(config.api_docs_url, timeout=10)
        response.raise_for_status()
        content = response.text
        return [Document(
            page_content=content,
            metadata={"source": config.api_docs_url, "type": "api_docs", "internal": True}
        )]
    except Exception as e:
        print(f"⚠️ Could not fetch API docs: {e}")
        return []


def ingest():
    all_docs = []

    loaders = [
        DirectoryLoader(config.data_path, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True),
        DirectoryLoader(config.data_path, glob="**/*.pdf", loader_cls=PyMuPDFLoader, silent_errors=True),
        DirectoryLoader(config.data_path, glob="**/*.csv", loader_cls=CSVLoader, silent_errors=True),
    ]

    for loader in loaders:
        try:
            all_docs += loader.load()
        except Exception:
            pass

    # Fetch API docs (internal use only)
    api_docs = fetch_api_docs()
    all_docs += api_docs

    if not all_docs:
        print("⚠️ No documents found.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    embeddings = OllamaEmbeddings(model=config.embedding_model)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(config.vectorstore_path)

    print(f"✅ Ingestion complete — {len(chunks)} chunks from {len(all_docs)} documents.")
