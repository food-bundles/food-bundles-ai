from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredXMLLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.config import config
from PIL import Image
import pytesseract
import requests
import glob
import os

# Set Tesseract path for Windows if not in PATH
if os.name == "nt":
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\scoop\apps\tesseract\current\tesseract.exe"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")


def load_images(data_path: str) -> list:
    """Extract text from images using OCR and return as Documents. Results are cached."""
    cache_file = os.path.join(data_path, ".ocr_cache.json")
    cache = {}

    # Load existing cache
    if os.path.exists(cache_file):
        import json
        with open(cache_file, "r") as f:
            cache = json.load(f)

    docs = []
    updated = False

    for ext in IMAGE_EXTENSIONS:
        for img_path in glob.glob(os.path.join(data_path, "**", ext), recursive=True):
            mtime = str(os.path.getmtime(img_path))
            cache_key = img_path

            # Use cached result if file hasn't changed
            if cache_key in cache and cache[cache_key]["mtime"] == mtime:
                text = cache[cache_key]["text"]
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": img_path, "type": "image_ocr"}
                    ))
                continue

            try:
                text = pytesseract.image_to_string(Image.open(img_path)).strip()
                cache[cache_key] = {"mtime": mtime, "text": text}
                updated = True
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": img_path, "type": "image_ocr"}
                    ))
                    print(f"📷 OCR extracted from: {os.path.basename(img_path)}")
                else:
                    print(f"⚠️ No text found in image: {os.path.basename(img_path)}")
            except Exception as e:
                print(f"⚠️ Could not process image {os.path.basename(img_path)}: {e}")

    # Save updated cache
    if updated:
        import json
        with open(cache_file, "w") as f:
            json.dump(cache, f)

    return docs


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
        DirectoryLoader(config.data_path, glob="**/*.md", loader_cls=TextLoader, silent_errors=True),
        DirectoryLoader(config.data_path, glob="**/*.pdf", loader_cls=PyMuPDFLoader, silent_errors=True),
        DirectoryLoader(config.data_path, glob="**/*.csv", loader_cls=CSVLoader, silent_errors=True),
        DirectoryLoader(config.data_path, glob="**/*.docx", loader_cls=Docx2txtLoader, silent_errors=True),
        DirectoryLoader(config.data_path, glob="**/*.xlsx", loader_cls=UnstructuredExcelLoader, silent_errors=True),
        DirectoryLoader(config.data_path, glob="**/*.html", loader_cls=UnstructuredHTMLLoader, silent_errors=True),
        DirectoryLoader(config.data_path, glob="**/*.xml", loader_cls=UnstructuredXMLLoader, silent_errors=True),
        DirectoryLoader(config.data_path, glob="**/*.pptx", loader_cls=UnstructuredPowerPointLoader, silent_errors=True),
    ]

    for loader in loaders:
        try:
            all_docs += loader.load()
        except Exception:
            pass

    # Load images via OCR
    all_docs += load_images(config.data_path)

    # Fetch API docs (internal use only)
    api_docs = fetch_api_docs()
    all_docs += api_docs

    if not all_docs:
        print("⚠️ No documents found.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    embeddings = OllamaEmbeddings(model=config.embedding_model, base_url=config.ollama_host)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(config.vectorstore_path)

    print(f"✅ Ingestion complete — {len(chunks)} chunks from {len(all_docs)} documents.")
