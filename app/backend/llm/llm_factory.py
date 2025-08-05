from importlib import import_module
from pathlib import Path
from re import DEBUG
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import logging

from app.backend.vector_store.faiss_store import load_faiss_index
from app.backend.utils.notifications import notifier
from app.backend.config.config import current_config

# Initialize logging
prompt_logger = logging.getLogger("prompt_loader")
prompt_logger.setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

# Updated default prompts with more flexible wording
_DEFAULT_PROMPTS = {
    "qa": """Answer based on these documents:
{context}

Question: {question}
Provide a concise answer:""",
    
    "greeting": "Hello! How can I help with {domain} information?",
    
    "error": """I couldn't find relevant information in our {domain} documents. 
For further assistance, please contact the {domain} team."""
}

def debug_print(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

from pathlib import Path

# Then modify load_faiss_index():
def load_faiss_index(embedder, persist_path, search_kwargs=None):
    # Convert to Path object if it's a string
    persist_path = Path(persist_path) if isinstance(persist_path, str) else persist_path
    print(f"Final persist path: {persist_path.absolute()}")

def get_llm():
    """Main LLM for Q&A"""
    try:
        from app.backend.vector_store.faiss_store import load_faiss_index
        from app.backend.retriever.pdf.splitter import get_embedder
        from app.backend.utils.notifier_instance import notifier        
        # Actual implementation would need your retriever config
        retriever = load_faiss_index(
            embedder=get_embedder(),
            persist_path = str(Path("app/data/domains") / current_config.domain / "vectorstore")
        )
        return ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0,
            streaming=True
        )
    except FileNotFoundError as e:
        notifier.send_slack(f"Vectorstore corruption: {str(e)}")
        raise

def get_fast_llm():
    """Faster LLM for citation formatting"""
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.3,
        max_tokens=200
    )

def get_domain_prompt(prompt_name: str) -> str:
    """Safe prompt loader with improved domain handling"""
    try:
        # 1. Try to load domain prompts
        module = import_module(f"backend.config.profiles.{current_config.domain}.prompts")
        prompts = getattr(module, 'PROMPTS', {})



        module_path = f"backend/config/profiles/{current_config.domain}/prompts.py"
        debug_print(f"🐞 Looking for prompts at: {module_path}")
        
        debug_print(f"🐞 Found prompts: {list(prompts.keys())}")



        
        # 2. Get the requested prompt or fallback
        prompt = prompts.get(prompt_name, _DEFAULT_PROMPTS.get(prompt_name, ""))
        debug_print(f"🐞 Selected prompt (first 100 chars): {prompt[:100]}...")

        if not Path(f"backend/config/profiles/{current_config.domain}/prompts.py").exists():
            print(f"⚠️ Using default prompts for {current_config.domain}")
        # 3. Format with domain if placeholder exists
        if "{domain}" in prompt:            
            debug_print("🐞 Formatting prompt with domain")
            return prompt.format(domain=current_config.domain)
        return prompt
        
    except Exception as e:
        debug_print(f"❌ Prompt loading failed: {str(e)}")
        prompt_logger.warning(f"Using default '{prompt_name}' prompt: {str(e)}")
        default = _DEFAULT_PROMPTS.get(prompt_name, "")        
        debug_print(f"🐞 Using default prompt: {default[:100]}...")

        # Format default prompt if it contains domain placeholder
        if "{domain}" in default:
            return default.format(domain=current_config.domain)
        return default

def get_greeting_message() -> str:
    """Specialized function for greeting messages"""
    greeting = get_domain_prompt("greeting")
    
    # Final fallback if all else fails
    if not greeting.strip():
        return f"Welcome! Ask me about {current_config.domain} topics."
    return greeting