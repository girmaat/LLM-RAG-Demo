 Backend Logic

This folder contains all the backend components for the RAG-based internal document assistant.  
Each module here performs a specific task in the document → retrieval → LLM → answer pipeline.

---

## Purpose

The `backend/` directory separates the core logic of the app from the user interface (`Streamlit`, `React`, or `FastAPI`).  
This makes the project easier to maintain, extend, and swap technologies in the future (e.g. changing vector DBs or LLMs).

---

## Subfolders

| Folder          | Purpose |
|-----------------|---------|
| `vector_store/` | Manages creation, storage, and retrieval of embeddings using FAISS, ChromaDB, etc. |
| `llm/`          | Encapsulates all logic to initialize and invoke different LLMs (Ollama, OpenAI, HuggingFace) |
| `retriever/`    | Handles document loading (PDFs), chunking, embedding, and preparation for vector storage |
| `pipeline/`     | Implements complete LangChain or Haystack pipelines to run end-to-end RAG logic |
| `memory/`       | (Optional) Stores chat history for multi-turn Q&A sessions |
| `api/`          | (Optional) REST or GraphQL endpoints (if migrating to FastAPI or server-based UI) |
| `utils/`        | Helper utilities like logging, formatting, error handling, config management, etc. |

---

## Flow Overview

Typical document query flow: [Upload PDF] → [retriever/] → [vector_store/] → [pipeline/] → [llm/] → [Answer]

Optional memory/history goes through `memory/`, and API routes go through `api/`.

---

## Modularity Benefits

- Easy to swap FAISS for Chroma or Qdrant
- Plug in new LLMs without breaking UI code
- Switch from Streamlit to FastAPI+React later
- Enable unit tests for each module

---

## Development Tips

- Keep each folder self-contained
- Expose consistent methods like `get_llm()`, `create_retriever()`, `run_chain()`
- Use `store_factory.py`, `llm_factory.py` patterns to support multiple backends

---

## Related Docs

- See `/frontend/` for UI
- See `/tests/` for test scripts
- See `/data/` for uploaded documents
- See `/vectorstore/` for saved vector DBs

