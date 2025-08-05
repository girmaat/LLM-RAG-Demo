from app.backend.retriever.pdf.loader import load_pdf
from app.backend.retriever.pdf.splitter import split_into_chunks

def load_and_split_pdf(file_path: str):
    """Combined PDF loading and splitting"""
    docs = load_pdf(file_path)
    return split_into_chunks(docs)