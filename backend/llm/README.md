# LLM Backends

This module manages all supported large language model (LLM) integrations for the RAG assistant.  
It is designed to support multiple **backends** (Ollama, OpenAI, Hugging Face, etc.) and multiple **models** within each backend (e.g. LLaMA 2, Mistral, GPT-4, Gemma).

---

## Purpose

This folder wraps each LLM provider behind a consistent interface.  
It allows the rest of the app to dynamically choose:
- Which backend to use (e.g., Ollama or OpenAI)
- Which model to load (e.g., llama2 or mistral)

---

## Structure

| File               | Purpose |
|--------------------|---------|
| `ollama_llm.py`     | Loads local models via Ollama (e.g., LLaMA 2, Mistral, Gemma) |
| `openai_llm.py`     | Uses OpenAI API (e.g., GPT-3.5, GPT-4) |
| `huggingface_llm.py`| Loads local models via Transformers (e.g., Gemma, Phi-2) |
| `gpt4all_llm.py`    | (Optional) Integration with GPT4All local desktop engine |
| `command_r_llm.py`  | (Optional) For future integration with RAG-tuned models like Command-R |
| `lmstudio_llm.py`   | (Optional) For local models loaded via LM Studio |
| `llm_factory.py`    | Central dispatch to load the correct backend + model |
| `__init__.py`       | Makes this a Python module and optionally exposes `get_llm()` |
| `README.md`         | This file — documents LLM backend structure and usage |

---

## Usage

Use the factory in your pipeline:
python
from backend.llm.llm_factory import get_llm

llm = get_llm(backend="ollama", model_name="mistral")

All LLM loaders return a LangChain-compatible object with .invoke() support.

Adding a New Backend or Model

    Create a new file: e.g., claude_llm.py

    Add a load_model(model_name: str) function

    Update llm_factory.py to route it

## Testing

    Each backend file should include a basic .invoke("test prompt") example

    Validate that all supported model names are mapped correctly

    Confirm that LangChain wrappers are returned from each loader

## Related Modules

    retriever/: loads and chunks documents

    vector_store/: handles embedding + search

    pipeline/: connects retriever + LLM into RAG