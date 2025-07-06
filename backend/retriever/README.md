# Retriever Module

This module handles all logic related to preparing documents for vector search and retrieval.  
It includes loading, parsing, chunking, and combining with vector store logic to create retrievers used in RAG pipelines.

---

## Purpose

In a RAG system, we must:
1. Load and parse documents (e.g. PDFs)
2. Split large text into smaller chunks
3. Embed and index these chunks for similarity search
4. Return relevant chunks during question answering

This folder centralizes that functionality for clean reuse across pipelines.

---

## Structure

| File                    | Purpose |
|-------------------------|---------|
| `pdf_loader.py`         | Loads PDF files using `pypdf` or alternative loaders |
| `splitter.py`           | Splits raw text into manageable overlapping chunks |
| `retriever_builder.py`  | Orchestrates loader, splitter, and vector store to return a retriever |
| `__init__.py`           | Makes this folder a Python module |

---

## Flow
[PDF Uploads] → [pdf_loader] → [splitter] → [vector_store] → retriever


Used by LangChain or Haystack pipelines for document question answering.

---

## Swappable Components

- Loader: `pypdf`, `pdfplumber`, `fitz`, etc.
- Splitter: Recursive, token-based, sentence-based
- Vector store: FAISS, Chroma, Qdrant

Each component is modular and configured independently for flexibility.

---

## Testing

Unit tests can be written to check:
- PDF text extraction from different loaders
- Chunk generation lengths and overlaps
- Retriever recall behavior

---

## Best Practices

- Keep loader logic independent of splitter logic
- Return `LangChain`-compatible document objects
- Use factory functions to simplify swaps (e.g. loader_type, chunk_size)
