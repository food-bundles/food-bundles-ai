import os
import shutil
import streamlit as st
from agents import run_agent
from ingestion import ingest

st.set_page_config(page_title="FoodBundles AI Agent", page_icon="🍱", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/meal.png", width=64)
    st.title("FoodBundles AI")
    st.caption("Powered by Ollama + RAG")
    st.divider()

    st.subheader("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload .txt files to the knowledge base",
        type=["txt", "pdf", "csv"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        saved = []
        for f in uploaded_files:
            dest = os.path.join(data_dir, f.name)
            with open(dest, "wb") as out:
                out.write(f.read())
            saved.append(f.name)

        if st.button("⚙️ Re-ingest Knowledge Base"):
            with st.spinner("Ingesting documents..."):
                ingest()
            st.success(f"✅ Ingested: {', '.join(saved)}")

    st.divider()
    st.caption("Documents stored in `/data` are used for RAG retrieval.")

# --- Main Chat UI ---
st.title("🍱 FoodBundles AI Assistant")
st.caption("Ask me anything about FoodBundles — ordering, payments, trader accounts, and more.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about FoodBundles..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = run_agent(prompt, st.session_state.messages)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
