# Tool Alternatives Overview

This document provides a summary of alternatives for the core tools used in this RAG-based document assistant project.  
It helps evaluate future migrations or scaling paths based on use case and deployment needs.

| Tool            | Current Choice     | Alternatives                     | Use Cases
|-----------------|--------------------|----------------------------------|------------------------------------------------------
| RAG Framework   | langchain          | llama-index, haystack            | LangChain: modular; LlamaIndex: simple RAG; Haystack: enterprise NLP |
| UI Framework    | streamlit          | gradio, flask, fastapi + react   | Streamlit: fast prototyping; FastAPI: scalable APIs                 |
| Vector Store    | faiss-cpu          | chromadb, qdrant, weaviate       | FAISS: local, fast; Qdrant: scalable, filterable; Chroma: persistent|
| PDF Loader      | pypdf              | pdfplumber, fitz, tika           | pypdf: fast text; pdfplumber: tables; tika: scanned/OCR              |
| Env Mgmt        | python-dotenv      | configparser, os.environ, pydantic | dotenv: simple; pydantic: validation; configparser: traditional ini |

All alternatives are pluggable with minimal changes if your architecture uses modular wrappers (recommended).
