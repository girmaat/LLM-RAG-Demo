import os
from langchain_community.vectorstores import FAISS

def build_faiss_index(chunks, embedder, persist_path=None):
    """Create a FAISS vector store from document chunks and optionally save to disk."""
    vectorstore = FAISS.from_documents(chunks, embedder)
    if persist_path:
        vectorstore.save_local(persist_path)
    return vectorstore

def load_faiss_index(persist_path, embedder):
    """Load a FAISS vector store from disk."""
    if os.path.exists(persist_path):
        return FAISS.load_local(persist_path, embeddings=embedder, allow_dangerous_deserialization=True)
    else:
        raise FileNotFoundError(f"FAISS index not found at {persist_path}")
