from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document
import os
from app.backend.config.config import current_config

def load_pdf(file_path: str) -> list[Document]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    abs_path = Path(file_path).absolute()
    filename = abs_path.stem  # This removes .pdf extension
    docs = []
    
    try:
        reader = PdfReader(file_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            'domain': current_config.domain,
                            'filename': filename + '.pdf',  # Explicitly add .pdf
                            'filepath': str(abs_path),
                            'page_number': page_num,
                            'doc_id': f"{filename}_{page_num}",
                            'total_pages': len(reader.pages)
                        }
                    )
                )
                
        if not docs:
            raise ValueError(f"No readable content in PDF: {file_path}")
            
        print(f"✅ Loaded {len(docs)} pages from {filename}.pdf")
        return docs
        
    except Exception as e:
        raise RuntimeError(f"Failed to load PDF {file_path}: {str(e)}") from e