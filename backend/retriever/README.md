# Retriever Module

This module handles all logic related to **data ingestion, preprocessing, and retrieval** from multiple content sources.  
It supports modular pipelines for PDFs, web pages, and APIs — all pluggable into your RAG backend via a common dispatcher.

---

## Purpose

The `retriever/` directory serves as a **source-agnostic loader layer** in the RAG pipeline.  
It takes responsibility for:
1. Loading data from a specific source (PDF, Web, API)
2. Chunking and embedding that data
3. Returning a retriever object that can be queried by the LLM pipeline

---

## Structure

| Folder/File           | Purpose |
|------------------------|---------|
| `pdf/`                 | Handles PDF-based loading and chunking (`pypdf`, `pdfplumber`, etc.) |
| `web/`                 | Handles web scraping and HTML parsing (e.g. `requests`, `BeautifulSoup`) |
| `api/`                 | Retrieves data from external APIs (e.g. JSON knowledge bases) |
| `dispatcher.py`        | Central router to choose data source(s) and initialize retriever |
| `__init__.py`          | Makes `retriever` importable as a module and optionally exposes dispatcher logic |

---

## Source Routing Flow

Streamlit Input → dispatcher.py
↳ pdf.retriever_builder
↳ web.retriever_builder
↳ api.retriever_builder
↓
returns Retriever

You can easily combine multiple sources in one call (e.g. load from PDF + API at once).

---

## How to Add a New Source

To support a new data source:
1. Create a new folder inside `retriever/` (e.g., `db/`, `excel/`, `md/`)
2. Add a `loader.py` and `retriever_builder.py` with consistent function signatures
3. Update `dispatcher.py` to call the new module conditionally

---

## Testing

Write tests for:
- File loading logic (PDFs, HTML, JSON, etc.)
- Chunk size + overlap behavior
- Embedding pipeline and retriever output

---

## Future Ideas

- Add hybrid retrieval from multiple sources
- Support user selection of loader backend (e.g. `pypdf` vs `pdfplumber`)
- Add metadata tagging (e.g. source type, confidence score)

---

## Related Modules

- `vector_store/`: Stores and retrieves embedded document chunks
- `llm/`: Generates answers based on retriever context
- `pipeline/`: Orchestrates the full RAG workflow
