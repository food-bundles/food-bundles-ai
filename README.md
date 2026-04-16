# Local AI Agent (Ollama)

## Install Ollama

https://ollama.ai

## Pull models

ollama pull llama3
ollama pull nomic-embed-text

## Install dependencies

pip install -r requirements.txt

## Ingest Data

python main.py ingest

## Query

python main.py query "Explain the system"

## Run UI

streamlit run streamlit_app.py

## Run API

uvicorn api:app --reload
