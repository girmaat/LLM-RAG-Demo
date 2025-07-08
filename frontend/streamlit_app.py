import streamlit as st
import tempfile
import os
import shutil
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.pipeline.rag_chain import build_qa_chain
from backend.llm.ollama_llm import get_ollama_llm
from backend.vector_store.faiss_store import build_faiss_index, load_faiss_index
from backend.retriever.pdf.loader import load_pdf
from backend.retriever.pdf.splitter import split_into_chunks, get_embedder


# Set page configuration
st.set_page_config(page_title="Chat with your Documents", layout="wide")

# Set index path early
index_path = "faiss_index"

# Sidebar
st.sidebar.title("📂 Document Chat Assistant")
st.sidebar.markdown("Upload your PDF files here.")

# File uploader
uploads = st.sidebar.file_uploader(
    label="Upload documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# Stop if no uploads
if not uploads:
    st.sidebar.info("⬆️ Please upload one or more supported documents to continue.")
    st.stop()

# Temporary folder
temp_dir = tempfile.TemporaryDirectory()
docs = []

# Load and process files
for file in uploads:
    temp_path = os.path.join(temp_dir.name, file.name)
    with open(temp_path, "wb") as f:
        f.write(file.getvalue())

    loader = load_pdf(temp_path)
    docs.extend(loader)

# 🧹 Clear old index and chat history on new upload
if os.path.exists(index_path):
    shutil.rmtree(index_path)
    st.write("⚠️ Cleared previous FAISS index due to new upload.")
    st.session_state.chat_history = []

# Split into chunks
chunks = split_into_chunks(docs)

# Embeddings
embedding_model = get_embedder()
st.write("🔄 Generating embeddings...")

# FAISS index (load or build)
try:
    vectorstore = load_faiss_index(index_path, embedding_model)
    st.write("✅ Loaded FAISS index from disk")
except FileNotFoundError:
    st.write("⚙️ Building new FAISS index...")
    vectorstore = build_faiss_index(chunks, embedding_model, persist_path=index_path)

# Retriever and LLM
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
llm = get_ollama_llm("llama2")
qa_chain = build_qa_chain(llm, retriever)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat UI
st.header("📘 Ask a Question About Your Document")

query = st.text_input("Ask a question:", key="user_input", placeholder="e.g. What is the company travel policy?")

if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []

# Handle question
if query:
    with st.spinner("Thinking..."):
        response = qa_chain.invoke(query)
        st.session_state.chat_history.append({
            "user": query,
            "bot": response["result"],
            "sources": response["source_documents"]
        })

# Show conversation
for i, turn in enumerate(st.session_state.chat_history):
    st.markdown(f"**🧑 You:** {turn['user']}")
    st.markdown(f"**🤖 Assistant:** {turn['bot']}")
    with st.expander(f"📄 Sources used for Question {i+1}"):
        for j, doc in enumerate(turn["sources"]):
            st.markdown(f"**Source {j+1}**")
            st.write(doc.page_content[:500])
