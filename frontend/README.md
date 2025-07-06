# Frontend

This folder contains all user interface components of the internal RAG document assistant.  
It currently supports a **Streamlit-based UI** and is structured to allow migration to a **React + API** setup if needed.

---

## Purpose

The `frontend/` directory holds all code that interacts directly with the user, including:
- File upload UI
- Question input interface
- Display of model responses
- (Optionally) future web-based React UI

---

## Structure

| File/Folder        | Purpose |
|--------------------|---------|
| `streamlit_app.py` | Current Streamlit application (used in MVP) |
| `react/`           | Placeholder for future React frontend, if migrating from Streamlit |
| `README.md`        | This documentation file |

---

## Usage

### Run the Streamlit App

From the project root (while in the correct Conda environment):

bash
streamlit run frontend/streamlit_app.py
Make sure:

    The backend modules (retriever, pipeline, vector_store) are available

    Uploaded PDFs go into the /data/ directory

### Future Migration Path

If you switch from Streamlit to a React-based frontend, the layout already supports:
frontend/
├── react/               # Frontend components (React, Vite, Next.js, etc.)
│   ├── src/
│   ├── public/
│   └── package.json
├── streamlit_app.py     # Legacy UI (optional)
'

##  UI Swap Strategy
UI Framework	Use Case	Status
Streamlit	MVP, internal testing, demos	✅ Active
React	Scalable production frontend	🚧 Planned

## Testing

You can create test files in tests/frontend/ for:

    Component rendering (React)

    File upload flow

    Response display

## Related Docs

    See /backend/ for all LLM and RAG logic

    See /data/ for uploaded PDFs

    See /vectorstore/ for vector DBs