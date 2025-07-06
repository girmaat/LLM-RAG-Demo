# LLM Backends

This module manages connections to various large language models (LLMs) supported in the RAG assistant.  
Each file here wraps a specific backend (Ollama, OpenAI, Hugging Face, etc.) so the rest of the app can call `get_llm()` without knowing the underlying engine.

---

## Purpose

This folder isolates all LLM logic from the rest of the backend.  
You can easily swap or add new models without touching the core RAG pipeline or UI code.

---

## Structure

| File               | Role |
|--------------------|------|
| `ollama_llm.py`     | Uses `Ollama(model="llama2")` via LangChain to run local models |
| `openai_llm.py`     | Connects to OpenAI’s cloud API (e.g. GPT-3.5, GPT-4) |
| `huggingface_llm.py` | Runs Hugging Face Transformers locally using `AutoModel` and `AutoTokenizer` |
| `llm_factory.py`    | Chooses which backend to use dynamically at runtime |
| `__init__.py`       | Makes this folder importable and optionally exposes a public API like `get_llm()` |

---

## Example Use

In the pipeline:
python
from backend.llm.llm_factory import get_llm

llm = get_llm("ollama")  # or "openai", "huggingface"
Each backend returns a LangChain-compatible LLM object with .invoke() support.

# Adding New Backends

To support another engine (e.g. GPT4All, LM Studio, Claude):

    Create gpt4all_llm.py or claude_llm.py

    Write a load_model() function that returns a compatible object

    Add a conditional to llm_factory.py

# Testing

Unit tests should:

    Mock each model’s invoke() method

    Ensure llm_factory.get_llm("backend") returns expected type

    Verify model configuration options (model name, temperature, etc.)

# Design Philosophy

    This folder hides LLM implementation details behind a consistent interface

    You can change LLMs without changing pipeline or UI logic

    Each file handles exactly one integration to simplify testing and debugging

# Related Modules

    retriever/: loads and splits documents

    vector_store/: embeds and stores text

    pipeline/: connects LLM and retriever to form a RAG system