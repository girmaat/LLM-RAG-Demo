# Pipeline: LLM Chains

This folder contains the core **RAG chain implementations** used in the project. These chains orchestrate how the language model, retriever, prompt, and output interact to generate answers from documents.

---

## Files in This Folder

### 1. `qa_chain.py`
- **Purpose:** Uses `RetrievalQA` from LangChain.
- **Approach:** High-level chain abstraction for ease of use.
- **Input:** Just the user query (a string).
- **Prompt Template Input Variables:** `context`, `query`

python
chain = RetrievalQA.from_chain_type(...)

### 2. lcel_chain.py

    - Purpose: LCEL (LangChain Expression Language)-style custom pipeline.
    - Approach: Fine-grained control using RunnableMap, RunnableLambda.
    - Input: A dictionary: {"query": "your question"}
    - Prompt Template Input Variables: context, query
chain = format_input | merge_prompt | llm | parse_output

## Switching Between Pipelines
In frontend/streamlit_app.py, control which chain to use by toggling the MODE variable: 
MODE = "retrievalqa"  # or "lcel"