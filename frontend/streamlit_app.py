import streamlit as st
import tempfile
import os
import shutil
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set this flag to switch between chains
MODE = "lcel"  # switch between "lcel" and "retrievalqa" if needed

# Import common modules
from backend.llm.ollama_llm import get_ollama_llm
from backend.vector_store.faiss_store import build_faiss_index, load_faiss_index
from backend.retriever.pdf.loader import load_pdf
from backend.retriever.pdf.splitter import split_into_chunks, get_embedder
from backend.agents.wikipedia_agent import query_wikipedia_agent

# --- Streamlit UI ---
st.set_page_config(page_title="Chat with your Documents", layout="wide")
st.sidebar.markdown(f"**Current pipeline:** `{MODE.upper()}`")

index_path = "faiss_index"

# Sidebar upload
st.sidebar.title("📂 Document Chat Assistant")
st.sidebar.markdown("Upload your PDF files here.")

# Source selector (PDF or Wikipedia)
source_option = st.sidebar.selectbox(
    label="Data Source",
    options=["PDF", "Wikipedia"],
    index=0
)

uploads = st.sidebar.file_uploader(
    label="Upload documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# Exit if no upload (PDF mode only)
if source_option == "PDF" and not uploads:
    st.sidebar.info("⬆ Please upload one or more supported documents to continue.")
    st.stop()

# Track uploaded file names
current_files = [file.name for file in uploads] if uploads else []
if "previous_files" not in st.session_state:
    st.session_state.previous_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Reset FAISS and history if new files uploaded
if source_option == "PDF" and current_files != st.session_state.previous_files:
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
        st.write("Detected new upload. Previous FAISS index cleared.")
    st.session_state.chat_history = []
    st.session_state.previous_files = current_files

# Process uploaded files (PDF only)
if source_option == "PDF":
    temp_dir = tempfile.TemporaryDirectory()
    docs = []
    for file in uploads:
        temp_path = os.path.join(temp_dir.name, file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getvalue())
        docs.extend(load_pdf(temp_path))

    # Split and embed
    chunks = split_into_chunks(docs)
    embedding_model = get_embedder()

    try:
        vectorstore = load_faiss_index(index_path, embedding_model)
        st.write("Loaded FAISS index from disk")
    except FileNotFoundError:
        st.write("Building new FAISS index...")
        vectorstore = build_faiss_index(chunks, embedding_model, persist_path=index_path)

    # LLM + Retriever + Chain
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = get_ollama_llm("llama2")

    if MODE == "retrievalqa":
        from backend.pipeline.qa_chain import build_qa_chain
        qa_chain = build_qa_chain(llm, retriever)
    else:
        from backend.pipeline.lcel_chain import build_lcel_chain
        qa_chain = build_lcel_chain(llm, retriever)

# --- Chat Interface ---
st.header("📘 Ask a Question")

query = st.text_input("", key="user_input", placeholder="e.g. Search internal policies, HR rules, IT help...")

if st.button("Clear Chat"):
    st.session_state.chat_history = []

# Handle query
if query:
    with st.spinner("Thinking..."):
        if source_option == "PDF":
            if MODE == "retrievalqa":
                response = qa_chain.invoke(query)
            else:
                response = qa_chain.invoke({"query": query})
            answer = response["result"]
            sources = response["source_documents"]
            source_type = "pdf"
        else:
            answer = query_wikipedia_agent(query)
            sources = []
            source_type = "wikipedia"

        st.session_state.chat_history.append({
            "user": query,
            "bot": answer,
            "sources": sources,
            "source_type": source_type,
            "timestamp": datetime.now().strftime("%m/%d/%Y %I:%M %p")
        })

# Show chat history
for i, turn in enumerate(st.session_state.chat_history):
    timestamp = turn.get("timestamp", "[No time]")
    st.markdown(f"**🧑 You** [{timestamp}]: {turn['user']}")
    st.markdown(f"**🤖 Assistant:** {turn['bot']}")

    if turn["source_type"] == "pdf":
        with st.expander(f"Sources used for Question {i+1}"):
            for j, doc in enumerate(turn["sources"]):
                st.markdown(f"**Source {j+1}**")
                st.write(doc.page_content[:500])
    elif turn["source_type"] == "wikipedia":
        st.markdown(f"_Answer from Wikipedia_")
