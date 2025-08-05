import sys
from pathlib import Path
import traceback
from typing import List
import argparse
from langchain_core.documents import Document
import os

from app.backend.config.config import current_config
# Debug flag
DEBUG = True

def debug_print(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)



# Add project root to Python path (two levels up from this file)


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
print(f"PROJECT_ROOT: {sys.path}")
sys.path.insert(0, str(PROJECT_ROOT))

# Verify the path was added
print(f"Python path: {sys.path}")

try:
    # Now import backend modules using absolute path
    from app.backend.retriever.pdf.loader import load_pdf
    from app.backend.retriever.pdf.splitter import split_into_chunks, get_embedder
    from app.backend.vector_store.faiss_store import build_faiss_index
    from app.backend.domains.manager import DomainManager
    print("All imports successful!")
except ImportError as e:
    print(f"Import failed: {e}")
    print("Current working directory:", os.getcwd())
    print("Project root:", PROJECT_ROOT)
    print("Backend exists:", (PROJECT_ROOT / 'backend').exists())
    raise


def process_pdf(file_path: str, domain: str) -> Path:
    """Process a single PDF file into the specified domain's vectorstore"""
    print(f"\n=== DEBUG: Starting PDF Processing ===")
    print(f"Input file: {file_path}")
    print(f"File exists: {Path(file_path).exists()}")
    
    try:
        # Set the domain first
        DomainManager.switch_domain(domain)
        print(f"Current domain: {current_config.domain}")
        
        output_dir = Path(f"app/data/domains/{domain}/vectorstore").absolute()
        print(f"Output directory: {output_dir}")
        print(f"Output parent exists: {output_dir.parent.exists()}")
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir.exists()}")
        
        print("Loading PDF...")
        docs = load_pdf(file_path)
        print(f"Loaded {len(docs)} pages")
        
        print("Splitting into chunks...")
        chunks = split_into_chunks(docs)
        print(f"Created {len(chunks)} chunks")
        
        print("Building FAISS index...")
        embedder = get_embedder()
        print(f"Using embedder: {embedder.model_name}")
        
        build_faiss_index(chunks, embedder, domain_name=current_config.domain)
        print("FAISS index built successfully")
        
        # Verify files were created
        print("\n=== Verifying Output Files ===")
        print(f"index.faiss exists: {(output_dir / 'index.faiss').exists()}")
        print(f"index.pkl exists: {(output_dir / 'index.pkl').exists()}")
        
        return output_dir
    except Exception as e:
        print(f"\n!!! Error processing {file_path}: {str(e)}")
        traceback.print_exc()
        raise

def process_all_pdfs(domain: str):
    """Process all PDFs in a domain directory"""
    debug_print(f"\n{'='*50}")
    debug_print(f"Starting processing for domain: {domain}")
    debug_print(f"{'='*50}\n")
    
    try:
        pdf_dir = Path(f"data/domains/{domain}")
        debug_print(f"Looking for PDFs in: {pdf_dir}")
        
        if not pdf_dir.exists():
            raise FileNotFoundError(f"Domain directory not found: {pdf_dir}")
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        debug_print(f"Found {len(pdf_files)} PDF files")
        
        for i, pdf_file in enumerate(pdf_files, 1):
            debug_print(f"\nProcessing file {i}/{len(pdf_files)}: {pdf_file.name}")
            try:
                process_pdf(str(pdf_file), domain)  # Pass domain parameter here
                print(f"✅ Processed {pdf_file.name}")
            except Exception as e:
                print(f"⚠️ Failed to process {pdf_file.name}: {str(e)}")
    except Exception as e:
        debug_print(f"Fatal error in process_all_pdfs: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDFs for a domain")
    parser.add_argument("--domain", required=True, help="Domain name (e.g., 'hr', 'finance')")
    args = parser.parse_args()
    
    try:
        debug_print("Script started")
        process_all_pdfs(args.domain)
    except Exception as e:
        debug_print("MAIN ERROR:", str(e))
        raise