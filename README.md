# <div align="center">🍱 FOODBUNDLES AI AGENT</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />

  <p align="center" class="tagline">
    <em>Intelligent. Contextual. Secure.</em><br>
    <strong>A fully local RAG-powered AI assistant for the FoodBundles food ordering and trading platform</strong>
  </p>
</div>

<div align="center">
  <a href="#features">
    <img src="https://img.shields.io/badge/✓-RAG_Pipeline-blue" alt="RAG Pipeline" />
  </a>
  <a href="#features">
    <img src="https://img.shields.io/badge/✓-Conversational_Memory-orange" alt="Conversational Memory" />
  </a>
  <a href="#features">
    <img src="https://img.shields.io/badge/✓-Live_API_Integration-green" alt="Live API Integration" />
  </a>
  <a href="#features">
    <img src="https://img.shields.io/badge/✓-Security_Rules-red" alt="Security Rules" />
  </a>
  <a href="#features">
    <img src="https://img.shields.io/badge/✓-Document_Upload-purple" alt="Document Upload" />
  </a>
</div>

## 📑 Table of Contents

- [👀 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [🛠️ Tech Stack](#️-tech-stack)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [🐳 Docker Setup](#-docker-setup)
- [🖥️ Usage](#️-usage)
- [🔐 Security](#-security)
- [🤝 Contributing](#-contributing)
- [📞 Contact](#-contact)

## 👀 Project Overview

FoodBundles AI Agent is a fully local, privacy-first AI assistant built for the [FoodBundles](https://food.rw) platform. It uses **Retrieval-Augmented Generation (RAG)** to answer questions based on platform documentation, live API data, and uploaded knowledge base files — all running locally with no cloud dependency.

The agent is powered by **Llama 3.2** via Ollama and uses **FAISS** for vector similarity search. It features a conversational Streamlit UI, a FastAPI REST endpoint, and strict security rules to prevent exposure of internal or sensitive data.

Key capabilities:

- **RAG Pipeline**: Retrieves relevant context from documents before answering
- **Live API Integration**: Automatically discovers and calls public API endpoints when documents lack information
- **Conversational Memory**: Maintains full chat history for contextual follow-up questions
- **Security Layer**: Blocks sensitive queries and never exposes internal API details to users
- **Document Upload**: Upload new `.txt`, `.pdf`, or `.csv` files via the UI to expand the knowledge base

## ✨ Key Features

### 🤖 Intelligent Query Routing

The agent automatically routes queries to the right handler:

- **General questions** (greetings, math, analysis) → answered directly by the LLM
- **FoodBundles questions** → retrieves context from the vector store first
- **Sensitive queries** (API keys, database, endpoints) → blocked immediately

### 📚 RAG Pipeline

Document-grounded responses using:

- Multi-format document ingestion (`.txt`, `.pdf`, `.csv`)
- Automatic API documentation fetching at ingest time
- FAISS vector store for fast similarity search
- Context injection into LLM prompts

### 🌐 Live API Integration

When documents don't have enough information:

- Scans OpenAPI/Swagger docs for a matching public endpoint
- Calls the endpoint live and injects real data into the response
- Never exposes endpoint URLs or internal details to users

### 💬 Conversational Memory

- Full chat history passed on every request
- Context-aware follow-up understanding
- Session-based memory in the Streamlit UI

### 🔐 Security Rules

- Sensitive query blocker (API keys, database, swagger, tokens)
- Internal API docs filtered from user-facing responses
- System prompt enforces strict data boundaries
- Users always redirected to [food.rw](https://food.rw)

### 📂 Document Upload UI

- Drag & drop file upload in the sidebar
- Files saved directly to `/data` folder
- One-click re-ingestion to update the knowledge base

## 🛠️ Tech Stack

- **LLM**: Llama 3.2 via [Ollama](https://ollama.ai) (fully local)
- **Embeddings**: `nomic-embed-text` via Ollama
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Framework**: LangChain + langchain-community
- **UI**: Streamlit
- **API**: FastAPI + Uvicorn
- **Document Loaders**: PyMuPDF, CSVLoader, TextLoader
- **Containerization**: Docker + Docker Compose

## 📁 Project Structure

<details>
<summary>📂 <b>View Complete Folder Structure</b></summary>

```
food-bundles-ai/
│
├── src/                        # Core source modules
│   ├── __init__.py
│   ├── agents.py               # Query routing, RAG, API integration
│   ├── config.py               # App configuration & environment
│   ├── ingestion.py            # Document ingestion pipeline
│   ├── retriever.py            # FAISS vector store retriever
│   ├── tools.py                # LangChain tools
│   └── memory.py               # Conversation memory
│
├── data/                       # Knowledge base documents
│   └── sample.txt              # FoodBundles platform info
│
├── vectorstore/                # FAISS index (auto-generated)
│   ├── index.faiss
│   └── index.pkl
│
├── main.py                     # CLI entry point
├── api.py                      # FastAPI REST endpoint
├── streamlit_app.py            # Streamlit chat UI
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image
├── docker-compose.yml          # Multi-service Docker setup
├── pull_models.sh              # Ollama model pull script
└── .gitignore
```

</details>

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running

### Installation

#### 1. Clone the repository

```bash
git clone https://github.com/EmmanuelSHYIRAMBERE/food-bundles-ai.git
cd food-bundles-ai
```

#### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Pull Ollama models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

#### 5. Ingest documents

```bash
python main.py ingest
```

#### 6. Run the UI

```bash
streamlit run streamlit_app.py
```

#### 7. Or run the API

```bash
uvicorn api:app --reload
```

## 🐳 Docker Setup

Run the full stack with a single command:

```bash
docker compose up --build
```

This starts:

| Service | URL | Description |
|---|---|---|
| `ollama` | `http://localhost:11434` | Local LLM server |
| `streamlit` | `http://localhost:8501` | Chat UI |
| `api` | `http://localhost:8000` | FastAPI REST endpoint |
| `ingest` | — | Runs once on startup |

To pull models inside the Ollama container:

```bash
docker exec -it foodbundles_ollama bash pull_models.sh
```

## 🖥️ Usage

### Chat UI

Visit `http://localhost:8501` to:

- Ask questions about FoodBundles features
- Upload new documents to expand the knowledge base
- Click **Re-ingest** after uploading to update the AI

### CLI

```bash
# Ingest documents
python main.py ingest

# Query from terminal
python main.py query "How do I place an order?"
python main.py query "What is Order Now Pay Later?"
```

### REST API

```bash
GET http://localhost:8000/query?q=How do I register as a trader?
```

Response:
```json
{
  "response": "To register as a trader on FoodBundles..."
}
```

## 🔐 Security

- Internal API documentation (`https://server.food.rw/api-docs`) is used only for knowledge — never exposed to users
- Sensitive queries (API keys, database, tokens, swagger) are blocked before reaching the LLM
- All user-facing responses reference [https://food.rw](https://food.rw) for platform actions
- Internal documents are tagged and filtered from user-visible context

## 🤝 Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some amazing feature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Contact

Emmanuel SHYIRAMBERE — [LinkedIn](https://www.linkedin.com/in/emashyirambere)

<div class="contact-section">
  <p align="center">
    <a href="mailto:emashyirambere1@gmail.com">Mail</a> |
    <a href="https://github.com/EmmanuelSHYIRAMBERE">GitHub</a> |
    <a href="https://food.rw">FoodBundles</a>
  </p>
</div>

<div align="center">
  <a href="#">
    <img src="https://img.shields.io/badge/↑-Back_to_Top-blue" alt="Back to Top" />
  </a>
</div>
