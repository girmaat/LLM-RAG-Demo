import sys
from pathlib import Path
import click

# Set the ABSOLUTE path to your project root
PROJECT_ROOT = Path(r"C:\Personal Folder\AI Studies\AI Projects\LLM-KM\AI-LLM-RAG")
sys.path.insert(0, str(PROJECT_ROOT))

# Import after setting path
from app.backend.config.config import current_config
from app.backend.pipeline.preprocess import process_pdf
from app.backend.domains.manager import DomainManager

@click.group()
def cli():
    pass

@cli.command()
@click.argument('file_path')
@click.option('--domain', required=True)
def process(file_path, domain):
    """Process a PDF into the specified domain's vectorstore"""
    try:
        print(f"\n=== DEBUG: Starting processing ===")
        print(f"🐞 Raw file_path input: {file_path}")
        
        abs_path = Path(file_path).absolute()
        print(f"🐞 Absolute file path: {abs_path} (exists: {abs_path.exists()})")
        # Convert domain to lowercase for case-insensitive comparison
        domain = domain.lower()
        
        # Manually validate domain first
        valid_domains = ["hr", "finance", "it"]  # Add all your valid domains
        if domain not in valid_domains:
            raise ValueError(f"Invalid domain. Must be one of: {valid_domains}")
            
        # Verify domain folder exists
        domain_path = Path("app/data/domains") / domain
        if not domain_path.exists():
            raise ValueError(f"Domain folder not found: {domain_path}")
            
        DomainManager.switch_domain(domain)
        output_path = process_pdf(file_path, domain)
        click.echo(f"✅ Processed {Path(file_path).name} → {output_path}")
    except Exception as e:
        click.secho(f"❌ Error: {str(e)}", fg='red')