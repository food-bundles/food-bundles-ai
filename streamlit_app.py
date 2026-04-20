import os
import shutil
import streamlit as st
from src.agents import run_agent
from src.ingestion import ingest

st.set_page_config(page_title="FoodBundles Admin — AI Agent", page_icon="⚙️", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.image("https://res.cloudinary.com/dzxyelclu/image/upload/v1760111270/Food_bundle_logo_cfsnsw.png", width=64)
    st.title("FoodBundles AI")
    st.divider()

    st.subheader("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload new files (.txt, .md, .pdf, .png, etc.) to the knowledge base",
        type=["txt", "md", "pdf", "csv", "docx", "xlsx", "html", "xml", "pptx", "jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        accept_multiple_files=True,
        key=st.session_state.get("uploader_key", "uploader_0"),
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
            # Reset uploader by changing its key
            st.session_state["uploader_key"] = f"uploader_{len(saved)}_{saved[0]}"
            st.rerun()

    st.divider()

# --- Main Chat UI ---
st.title("⚙️ FoodBundles Admin — AI Agent")
st.caption("Admin panel for managing the knowledge base and testing the AI agent.")

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
