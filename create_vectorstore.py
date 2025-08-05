import sys
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).parent))

from app.backend.pipeline.preprocess import process_pdf
from app.backend.config.config import current_config
from app.backend.domains.manager import DomainManager

# Configure logging
logging.basicConfig(
    filename='vectorstore.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_last_processed_time(log_file='vectorstore.log'):
    """Get the last processing time from log file"""
    try:
        with open(log_file, 'r') as f:
            for line in reversed(list(f)):
                if 'Completed processing' in line:
                    return datetime.strptime(line.split(' - ')[0], '%Y-%m-%d %H:%M:%S,%f')
    except FileNotFoundError:
        return datetime.min
    return datetime.min

def process_pdf_wrapper(pdf_file, domain):
    """Wrapper function for parallel processing"""
    try:
        logger.info(f"Starting processing: {pdf_file.name}")
        start_time = datetime.now()
        
        output_dir = process_pdf(str(pdf_file), domain)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Completed processing {pdf_file.name} in {duration:.2f}s")
        return True, pdf_file.name
    except Exception as e:
        logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
        return False, pdf_file.name

def process_all_hr_pdfs():
    """Process all PDFs in the HR domain folder"""
    domain = "hr"
    DomainManager.switch_domain(domain)
    hr_pdf_dir = Path("app/data/domains/hr")
    vectorstore_dir = hr_pdf_dir / "vectorstore"

    # Validate paths
    if not hr_pdf_dir.exists():
        logger.error(f"HR directory not found at {hr_pdf_dir}")
        return False

    # Get PDFs modified since last run
    last_processed = get_last_processed_time()
    pdf_files = [f for f in hr_pdf_dir.glob("*.pdf") 
                if f.stat().st_mtime > last_processed.timestamp()]
    
    if not pdf_files:
        logger.info("No new or modified PDFs found since last run")
        return True

    logger.info(f"Found {len(pdf_files)} PDFs to process:")
    for pdf in pdf_files:
        logger.info(f"- {pdf.name} (modified: {datetime.fromtimestamp(pdf.stat().st_mtime)})")

    # Process in parallel (4 workers)
    success_count = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_pdf_wrapper, pdf, domain) 
                  for pdf in pdf_files]
        
        for future in as_completed(futures):
            success, filename = future.result()
            if success:
                success_count += 1
                print(f"✓ {filename}")
            else:
                print(f"✗ {filename}")

    logger.info(f"Processing complete. {success_count}/{len(pdf_files)} files succeeded")
    return success_count == len(pdf_files)

if __name__ == "__main__":
    print("=== Starting PDF Processing ===")
    if process_all_hr_pdfs():
        print("\n✅ All files processed successfully")
    else:
        print("\n⚠️ Some files failed to process (check vectorstore.log)")
    print("\nCheck 'vectorstore.log' for details")