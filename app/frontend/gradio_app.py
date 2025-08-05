import asyncio
import html
import os
import random
import re
import sys
from pathlib import Path
import time
import traceback
from typing import List, Iterator, Dict, Any
import gradio as gr
from langchain_core.documents import Document
from urllib.parse import quote

from app.backend.config.profiles.hr.prompts import COMPANY_NAME
PROJECT_ROOT = Path("/app")
sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.config.config import current_config
from app.backend.utils.notifier_instance import notifier
from app.backend.utils.notifications import NotificationService

from app.backend.llm.llm_factory import get_llm, get_domain_prompt
from app.backend.pipeline.qa_chain import build_qa_chain
from app.backend.vector_store.faiss_store import load_faiss_index
from app.backend.retriever.pdf.splitter import get_embedder
from app.backend.domains.validator import DomainValidator
from app.backend.retriever.dispatcher import ToolDispatcher
from app.backend.tools.tool_factory import get_tool
from dotenv import load_dotenv
from app.backend.utils.notification_validator import validate_notification_config


# --- Path Setup ---
sys.path.append(str(Path(__file__).parent.parent))

# --- Configuration ---


DEBUG = True
qa_chain = None
# Animation control constants
TYPING_SPEED = 0.02  # Seconds per character
CHUNK_SIZE = 2        # Characters per chunk
PUNCTUATION_DELAY = 0.15  # Extra delay after punctuation


VECTORSTORE_PATH = Path(__file__).parent.parent.parent / "app" / "data" / "domains" / "hr" / "vectorstore"
print(f"Resolved vectorstore path: {VECTORSTORE_PATH.absolute()}")

print(f"Project root: {Path(__file__).parent.parent.parent}")
print(f"PDF exists: {Path('app/data/domains/hr/EmployeeHandbook.pdf').exists()}")
print(f"Vectorstore exists: {Path('app/data/domains/hr/vectorstore').exists()}")



print(f"Vectorstore path: {VECTORSTORE_PATH}")
print(f"Vectorstore exists: {VECTORSTORE_PATH.exists()}")





def get_notifier():
    """Singleton pattern for notification service"""
    if not hasattr(get_notifier, "_instance"):
        get_notifier._instance = NotificationService()
    return get_notifier._instance

def initialize_application():
    from app.backend.domains.validator import DomainValidator
    is_valid, msg = DomainValidator.validate()
    if not is_valid:
        print(f"HR domain invalid: {msg}")
        raise RuntimeError(f"HR domain invalid: {msg}")

    """Initialize all components with proper error handling"""    
    # ===== Add Path Verification Here =====
    def verify_paths():
        required_paths = {
            "PDF": Path("app/data/domains/hr/EmployeeHandbook.pdf"),
            "Vectorstore": Path("app/data/domains/hr/vectorstore"),
            "FAISS index": Path("app/data/domains/hr/vectorstore/index.faiss"),
            "PKL metadata": Path("app/data/domains/hr/vectorstore/index.pkl")
        }
        
        print("\n=== Path Verification ===")
        for name, path in required_paths.items():
            print(f"{name}: {path.exists()} at {path.absolute()}")
        
        return all(path.exists() for path in required_paths.values())

    if not verify_paths():
        raise RuntimeError("Required files missing - check path verification above")
    # ===== End of Path Verification =====

    # Rest of the existing initialize_application() code...
    global qa_chain
    try:
        # 1. Validate domain configuration
        from app.backend.domains.validator import DomainValidator
        is_valid, msg = DomainValidator.validate()
        if not is_valid:
            error_msg = f"{current_config.domain.upper()} domain invalid: {msg}"
            debug_print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        debug_print(f"🔍 Initializing {current_config.domain.upper()} domain application")

        # 2. Initialize notification service
        debug_print("\n=== Initializing Notification Service ===")
        try:
            from app.backend.utils.notifications import notifier
            if notifier._enabled:
                debug_print("✅ Notification service is active")
                # Test notification
                notifier.send_pushover(
                    message="Application starting up",
                    title=f"{current_config.domain.upper()} System Startup"
                )
            else:
                debug_print(f"❌ Notification service disabled: {notifier._init_error or 'Unknown error'}")
        except Exception as e:
            debug_print(f"⚠️ Notification service initialization warning: {str(e)}")

        # 3. Validate notification configuration
        debug_print("\n=== Validating Notification Config ===")
        validate_notification_config()
        debug_print("✅ Notification config validated")

        # 4. Initialize core components
        debug_print("\n=== Initializing Components ===")
        debug_print(f"🐞 Domain: {current_config.domain}")
        debug_print(f"🐞 Vectorstore path: {VECTORSTORE_PATH}")
        
        qa_chain = initialize_components()
        debug_print("✅ Backend initialized successfully")
  
        return True
        
    except Exception as e:
        error_msg = f"Application initialization failed: {str(e)}"
        debug_print(f"\n❌ {error_msg}", exc_info=True)
        
        # Attempt error notification if possible
        if 'notifier' in locals() and getattr(notifier, '_enabled', False):
            try:
                notifier.send_pushover(
                    message=error_msg,
                    title="Application Startup Failed",
                    priority=1
                )
            except Exception as notify_error:
                debug_print(f"⚠️ Failed to send error notification: {notify_error}")
        
        raise RuntimeError(error_msg) from e

env_path = Path(__file__).parent.parent / ".env"
print(f"🛠️ Loading .env from: {env_path}")  # Debug path
load_dotenv(env_path)

# Debug print keys
print(f"PUSHOVER_API_KEY exists: {bool(os.getenv('PUSHOVER_API_KEY'))}")
print(f"PUSHOVER_USER_KEY exists: {bool(os.getenv('PUSHOVER_USER_KEY'))}")


# --- Utilities ---
def debug_print(*args, **kwargs):
    if DEBUG:
        # Remove exc_info handling from print
        message = " ".join(str(arg) for arg in args)
        if "exc_info" in kwargs:
            import traceback
            message += "\n" + "".join(traceback.format_exc())
        print("🐞 DEBUG:", message)

def safe_get_prompt(prompt_name: str, default: str) -> str:
    """Safely get domain prompt with fallback"""
    try:
        # Get base prompt
        prompt = get_domain_prompt(prompt_name) or default
        
        # Custom handling for greeting prompt
        if prompt_name == "greeting":
            prompt = prompt.rstrip('.!?')  # Clean existing punctuation
            prompt += ". Start typing in the box below..."
            
            # Add current domain if {domain} placeholder exists
            if "{domain}" in prompt:
                prompt = prompt.format(domain=current_config.domain)
        
        # Ensure non-empty result
        return prompt.strip() or default
        
    except Exception as e:
        debug_print(f"⚠️ Prompt '{prompt_name}' failed: {str(e)}")
        # Return formatted default if it contains {domain}
        if "{domain}" in default:
            return default.format(domain=current_config.domain)
        return default
        
def initialize_components():
    global qa_chain
    """Debugged initialization with comprehensive prompt checks"""
    try:
        debug_print("🚀 Initializing components...")
        print("\n=== PATH DEBUGGING ===")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python path: {sys.path}")

        # 1. Verify domain prompt file exists
        prompt_file = Path(f"app/backend/config/profiles/{current_config.domain}/prompts.py")
        
        debug_print(f"🐞 Prompt file path: {prompt_file}")
        debug_print(f"🐞 Prompt file exists: {prompt_file.exists()}")

        if not prompt_file.exists():
            debug_print(f"⚠️ Missing prompt file at {prompt_file}")
        
        # 2. Test prompt loading
        debug_print("🐞 Testing prompt loading...")
        test_prompt = get_domain_prompt("qa")        
        debug_print(f"🐞 Loaded QA prompt (first 100 chars): {test_prompt[:100]}...")

        if "{context}" not in test_prompt or "{question}" not in test_prompt:
            raise ValueError("QA prompt missing required placeholders")
        debug_print("✅ Prompt validation passed")

        # 3. Initialize retriever
        debug_print(f"🐞 Vectorstore path: {VECTORSTORE_PATH}")
        debug_print(f"🐞 Vectorstore exists: {VECTORSTORE_PATH.exists()}")
        debug_print(f"🐞 Contents of vectorstore: {list(VECTORSTORE_PATH.glob('*')) if VECTORSTORE_PATH.exists() else 'N/A'}")

        # Initialize retriever variable before conditional block
        retriever = None
        
        if not VECTORSTORE_PATH.exists():
            parent = VECTORSTORE_PATH.parent
            print(f"\nParent directory contents ({parent}):")
            try:
                print(os.listdir(parent))
            except Exception as e:
                print(f"Couldn't list parent directory: {str(e)}")
            raise RuntimeError(f"Vectorstore directory not found at {VECTORSTORE_PATH}")
        
        try:
            print(f"\nLoading FAISS index from: {VECTORSTORE_PATH}")
            print(f"Contents of vectorstore: {os.listdir(VECTORSTORE_PATH) if VECTORSTORE_PATH.exists() else 'DIRECTORY NOT FOUND'}")
            
            retriever = load_faiss_index(
                get_embedder(),
                str(VECTORSTORE_PATH),
                search_kwargs={'k': 3, 'score_threshold': 0.85}
            )
            print("✅ FAISS index loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load FAISS index: {str(e)}")
            raise

        # 4. Build QA chain with validation
        debug_print("🐞 Building QA chain...")

        if retriever is None:
            raise RuntimeError("Retriever was not properly initialized")

        qa_chain = build_qa_chain(
            llm=get_llm(),
            retriever=retriever,
            company_name=COMPANY_NAME
        )
        print("✅ QA chain built successfully")

        generate_response.qa_chain = qa_chain
        debug_print(f"🐞 QA chain stored: {qa_chain is not None}")
        
        return qa_chain

    except Exception as e:
        debug_print(f"❌ Component initialization failed: {str(e)}")
        raise RuntimeError(f"Component initialization failed: {str(e)}")

def format_response(response_data: Dict[str, Any]) -> str:
    """Format response with proper PDF citations and next step questions"""
    try:
        answer = str(response_data.get("answer", ""))
        sources = response_data.get("sources", [])
        
        # Add company reference 25% of the time
        if random.random() < 0.25 and sources:
            lead_phrases = [
                f"According to {COMPANY_NAME} policies,",
                f"As outlined in {COMPANY_NAME} documentation,",
                f"Per {COMPANY_NAME} guidelines,",
                f"{COMPANY_NAME} policy states that"
            ]
            answer = f"{random.choice(lead_phrases)} {answer[0].lower() + answer[1:]}"
        
        # Process citations
        citation_links = []
        for doc in sources:
            filename = doc.metadata.get('filename', '')
            page = doc.metadata.get('page_number', '')
            
            if filename and page:
                if not filename.lower().endswith('.pdf'):
                    filename += '.pdf'
                
                prefix = filename[:2].upper()
                pdf_url = f"/api/v1/pdf/open?filename={quote(filename)}&page={page}"
                
                citation_links.append(
                    f'<a href="{pdf_url}" target="_blank" '
                    f'class="page-link">{prefix}-{page}</a>'
                )
        
        # Add citations if available
        if citation_links:
            answer += f'<div class="citations">[Pages: {", ".join(citation_links)}]</div>'
        
        # Add a next step question (25% chance if no question exists)
        if "?" not in answer[-10:] and random.random() < 0.25:
            next_steps = [
                "Would you like me to clarify any part of this?",
                "Should I provide the full policy document?",
                "Can I help with anything else regarding this?",
                "Would you like me to connect you with HR for more details?"
            ]
            answer += f'<div class="next-step">\n{random.choice(next_steps)}</div>'
        
        return answer
        
    except Exception as e:
        print(f"⚠️ Error formatting response: {str(e)}")
        return str(response_data.get("answer", ""))  # Return basic answer if formatting fails

def generate_response(message: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """Generate response with proper input handling"""
    debug_print(f"\n=== NEW QUERY: {message} ===")
    
    try:
        qa_chain = getattr(generate_response, 'qa_chain', None)        
        if not qa_chain:
            return error_response("Document search system not initialized")

        # Format history if available
        chat_history_str = ""
        if chat_history:
            chat_history_str = "\n".join(
                f"{msg['role'].capitalize()}: {msg['content']}" 
                for msg in chat_history[-3:]
            )
        
        # Prepare inputs
        inputs = {
            "question": message,
            "chat_history": chat_history_str,
            "company_name": COMPANY_NAME
        }
        
        debug_print("🐞 Invoking QA chain with inputs:", inputs)
        
        # Get the response
        response = qa_chain(inputs)
        
        debug_print(f"🐞 Raw response: {str(response)[:200]}...")
        
        # Response handling
        answer = response.get("answer", "")
        sources = response.get("sources", [])
        
        # Format the answer
        answer_text = str(answer)
        
        # Add next step question if needed
        if not any(q in answer_text for q in ['?', 'clarify', 'help', 'details']):
            next_steps = [
                "\n\nWould you like me to clarify any part of this?",
                "\n\nShould I provide the full policy document?",
                "\n\nCan I help with anything else regarding this?",
                f"\n\nWould you like me to connect you with {COMPANY_NAME} HR for more details?"
            ]
            answer_text += random.choice(next_steps)

        return {
            "answer": answer_text,
            "sources": sources
        }

    except Exception as e:
        debug_print(f"❌ Query failed: {str(e)}", exc_info=True)
        return error_response("Error searching documents")

def error_response(message: str) -> Dict[str, Any]:
    """Standard error response"""
    return {
        "answer": f"Sorry, {message}",
        "sources": []
    }

async def chat_respond(message: str, history: List[List[str]]) -> Iterator[List[Dict[str, str]]]:
    try:
        print(f"\n=== DEBUG: Starting processing for query: {message} ===")  # Hardcoded print
        
        # Debug notification service status
        print(f"Notification service enabled: {notifier._enabled}")
        print(f"Detector initialized: {hasattr(notifier, 'detector')}")
        
        response = generate_response(message)
        print(f"Raw response: {response}")
        
        # Add this debug before notification
        print("=== DEBUG: Calling analyze_and_notify ===")
        await notifier.analyze_and_notify(message, response)
        print("=== DEBUG: Notification call completed ===")

        if isinstance(response, dict):
            await notifier.analyze_and_notify(
                query=message,
                response={
                    "answer": response.get("answer", ""),
                    "sources": response.get("sources", [])
                }
            )
        else:
            formatted = str(response)
        
        # Convert to Gradio's messages format
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": formatted}
        ]
        
        yield new_history
        
    except Exception as e:
        debug_print(f"⚠️ Chat respond error: {str(e)}")
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Sorry, I encountered an error processing your request."}
        ]
CSS = """
/* ===== BASE STYLES ===== */
html, body, .gradio-container, .gradio-app {
    height: 100vh !important;
    width: 100vw !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}

/* ===== MAIN LAYOUT ===== */
.gr-row {
    display: flex !important;
    height: 100vh !important;
    margin: 0 !important;
    overflow: hidden !important;
}

/* ===== LEFT PANEL ===== */
.left-panel {
    background: linear-gradient(135deg, #1a2a3a 0%, #1e3a5f 100%) !important;
    padding: 40px 25px !important;
    width: 300px !important;
    min-width: 300px !important;
    height: 100vh !important;
    overflow-y: auto !important;
    color: #f8fafc !important;
    box-shadow: 5px 0 15px rgba(0,0,0,0.15) !important;
    position: relative !important;
    z-index: 10 !important;
}

.left-panel::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 120px;
    background: linear-gradient(to bottom, rgba(255,255,255,0.1) 0%, transparent 100%);
}

.profile-image-container {
    position: relative;
    margin-bottom: 30px !important;
}

.profile-image {
    width: 140px;
    height: 140px;
    border-radius: 50%;
    object-fit: cover;
    margin: 0 auto;
    display: block;
    /* Modern shadow effect */
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    /* Smooth hover effect */
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.profile-image:hover {
    transform: scale(1.03) !important;
    transform: scale(1.05);
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
}

.profile-name {
    color: white !important;
    font-size: 1.7rem !important;
    font-weight: 600 !important;
    margin-bottom: 5px !important;
    text-align: center !important;
    letter-spacing: 0.5px !important;
    text-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
}

.profile-title {
    color: #cbd5e1 !important;
    font-size: 1rem !important;
    text-align: center !important;
    margin-bottom: 30px !important;
    font-weight: 400 !important;
    letter-spacing: 0.3px !important;
}

.profile-description {
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    margin-bottom: 30px !important;
    padding: 0 10px !important;
}

.download-btn {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
    color: white !important;
    border: none !important;
    padding: 12px 25px !important;
    border-radius: 24px !important;
    font-weight: 500 !important;
    margin: 25px auto !important;
    display: block !important;
    width: max-content !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 10px rgba(59, 130, 246, 0.3) !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px !important;
}

.download-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 15px rgba(59, 130, 246, 0.4) !important;
}

.divider-section {
    margin-top: 30px !important;
    padding-top: 30px !important;
    border-top: 1px solid rgba(255,255,255,0.1) !important;
}

.divider-title {
    color: #f8fafc !important;
    font-size: 1.1rem !important;
    font-weight: 500 !important;
    margin-bottom: 15px !important;
    text-align: center !important;
    letter-spacing: 0.5px !important;
}

.divider-text {
    color: #cbd5e1 !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
    margin-bottom: 20px !important;
    text-align: center !important;
}

.blog-link {
    color: #93c5fd !important;
    text-decoration: none !important;
    font-weight: 500 !important;
    display: block !important;
    text-align: center !important;
    margin: 15px 0 !important;
    transition: color 0.2s ease !important;
}

.blog-link:hover {
    color: #60a5fa !important;
    text-decoration: underline !important;
}
/* ===== RIGHT PANEL ===== */
.right-column {
    display: flex !important;
    flex-direction: column !important;
    flex: 2 !important;
    height: 100vh !important;
    position: relative !important;
    overflow: hidden !important;
    background: #f5f7fa !important; /* Light gray-blue background */
}

/* Header */
.chat-header {
    padding: 20px 30px !important;
    background: white !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 99 !important;
    flex: none !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
    border-bottom: 1px solid #eaeef2 !important;
}

.chat-header h2 {
    margin: 0 !important;
    color: #2c3e50 !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
}

/* Chat Container */
.chat-scroll-container {
    flex: 1 !important;
    overflow-y: auto !important;
    display: flex !important;
    flex-direction: column !important;
    min-height: 0 !important;
    background: #f5f7fa !important;
    padding-bottom: 20px !important;
}

/* Message Bubbles */
.message {
    max-width: 90% !important;
    margin: 8px 30px !important;
    line-height: 1.5 !important;
    position: relative !important;
    word-wrap: break-word !important;
    border-radius: 18px !important;
}

.message.bot {
    background: #f8f8f8 !important;
    border-radius: 18px 18px 18px 4px !important;
    margin-right: auto !important;
    margin-left: 30px !important;
    border: 1px solid #e7e7e7 !important;
    color: #2c3e50 !important;
}

.message.user {
    background: #fef6df  !important;
    color: white !important;
    border-radius: 18px 18px 4px 18px !important;
    margin-right: 30px !important;
}




/* Citations */
.citations {
    color: #7f8c8d !important;
    font-size: 0.8em !important;
    margin-top: 8px !important;
}

.page-link {
    color: #3498db !important;
    text-decoration: none !important;
    font-weight: 500 !important;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {
    width: 8px !important;
}

::-webkit-scrollbar-track {
    background: #f1f1f1 !important;
    border-radius: 4px !important;
}

::-webkit-scrollbar-thumb {
    background: #c1c1c1 !important;
    border-radius: 4px !important;
}

::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8 !important;
}

/* ===== MOBILE RESPONSIVENESS ===== */
@media (max-width: 768px) {
    .gr-row {
        flex-direction: column !important;
    }
    
    .left-panel {
        width: 100% !important;
        height: auto !important;
        max-height: 40vh !important;
        padding: 20px 15px !important;
    }
    
    .right-column {
        height: auto !important;
        min-height: 60vh !important;
    }
    
    .message {
        max-width: 85% !important;
        margin-left: 15px !important;
        margin-right: 15px !important;
    }
}
/* ===== TEXTBOX & SUBMIT BUTTON STYLING ===== */
.textbox-wrapper {
    position: sticky !important;
    bottom: 0 !important;
    background: white !important;
    padding: 20px 30px !important;
    z-index: 100 !important;
    width: 100% !important;
    box-shadow: 0 -5px 30px rgba(0,0,0,0.12) !important;
    display: flex !important;
    gap: 12px !important;
    align-items: center !important;
    border-radius: 24px 24px 0 0 !important;
    border-top: 1px solid rgba(0,0,0,0.08) !important;
    border-left: 1px solid rgba(0,0,0,0.05) !important;
    border-right: 1px solid rgba(0,0,0,0.05) !important;
}

#question-input {
    border: 2px solid #e2e8f0 !important;
    border-radius: 24px !important;
    padding: 16px 24px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    font-size: 1rem !important;
    outline: none !important;
    transition: all 0.25s ease !important;
    background: white !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
    font-weight: 500 !important;
    color: #1e293b !important;
}

#question-input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 4px rgba(59,130,246,0.15), 
                0 2px 12px rgba(0,0,0,0.1) !important;
}

#question-input::placeholder {
    color: #94a3b8 !important;
    font-weight: 400 !important;
    opacity: 1 !important;
}

.submit-btn {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 50% !important;
    width: 48px !important;
    height: 48px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 5px rgba(59,130,246,0.3) !important;
    flex-shrink: 0 !important;
}

.submit-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 8px rgba(59,130,246,0.4) !important;
}

.submit-btn:active {
    transform: translateY(0) !important;
    box-shadow: 0 1px 3px rgba(59,130,246,0.3) !important;
}

.submit-icon {
    font-size: 1.2rem !important;
    margin-left: 2px !important; /* Adjust arrow position */

}

/* Desclaimer */
.disclaimer-text {
    color: #e2e8f0 !important;  /* Light gray-blue for visibility */
    font-size: 0.85rem !important;
    line-height: 1.5 !important;
    margin-top: 15px !important;
}

.disclaimer-text a {
    color: #93c5fd !important;  /* Maintains your link color */
    text-decoration: underline !important;
}

.disclaimer-text a:hover {
    color: #60a5fa !important;  /* Slightly brighter on hover */
}

"""

async def load_greeting():
    
    greeting = safe_get_prompt('greeting', 'How can I help?')
    
    # Remove HTML escaping and fix formatting
    greeting = greeting.replace("&#x27;", "'")  # Convert HTML entities back to normal characters
    greeting = greeting.replace("<br>", "\n")   # Convert <br> to newlines
    
    # Initial empty container with stable dimensions
    yield """
    <div class="greeting-message">
        <div class="greeting-stream"></div>
    </div>
    """
    await asyncio.sleep(0.3)  # Initial pause
    
    # Stream text in chunks
    displayed_text = ""
    for i in range(0, len(greeting), CHUNK_SIZE):
        chunk = greeting[i:i+CHUNK_SIZE]
        displayed_text += chunk
        
        # Calculate dynamic delay
        current_delay = TYPING_SPEED * CHUNK_SIZE
        if any(punct in chunk for punct in {'.', '!', '?'}):
            current_delay = PUNCTUATION_DELAY
        
        # Render with stable container
        yield f"""
        <div class="greeting-message">
            <div class="greeting-stream">
                {html.escape(displayed_text)}<span class="typing-cursor"></span>
            </div>
        </div>
        """
        await asyncio.sleep(current_delay)
    
    # Final render (no cursor)
    yield f"""
    <div class="greeting-message">
        <div class="greeting-stream">
            {html.escape(displayed_text)}
        </div>
    </div>
    """

def create_chat_interface():
    with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as interface:
        with gr.Row():
            # Left Profile Panel
            with gr.Column(scale=1, min_width=220, elem_classes="left-panel"):
                with gr.Column(elem_classes="profile-column"):
                    # Profile image
                    with gr.Column(elem_classes="profile-image-container"):
                        gr.Image(
                            value=current_config.personal["profile_image"],
                            elem_classes="profile-image",
                            show_label=False,
                            show_download_button=False
                        )
                    
                    # Profile content
                    with gr.Column(elem_classes="profile-content"):
                        gr.Markdown(f"""
                        <h1 class="profile-name">{current_config.personal['name']}</h1>
                        <h2 class="profile-title">{current_config.personal['title']}</h2>
                        <h2 class="profile-description">{current_config.personal['description']}</h2>
                        """)
                        
                        download_btn = gr.Button(
                            current_config.personal["resume_button_text"],
                            elem_classes="download-btn"
                        )
                        file_download = gr.File(visible=False)

                        with gr.Column(elem_classes="divider-section"):
                            disclaimer = current_config.ui_config.get("disclaimer", {})
                            gr.Markdown(f"""<h3 class="divider-title">{current_config.ui_config['titles']['assistant_title']}</h3>""")
                            gr.Markdown(f"""<a href="{current_config.personal['blog_url']}" target="_blank" class="blog-link">{current_config.ui_config['titles']['blog_link_text']}</a>""")
                            gr.Markdown(f"""<p class="divider-text">{current_config.ui_config['descriptions']['assistant_description']}</p>""")
                            gr.Markdown(f"""<h3 class="divider-title">{current_config.ui_config['titles']['start_chatting_title']}</h3>""")
                            gr.Markdown(f"""<p class="divider-text">{current_config.ui_config['chat']['start_chatting_text']}</p>""")
                            gr.Markdown(f"""
                                <div style="color: #e2e8f0; font-size: 0.85rem; line-height: 1.5;">
                                <strong style="color: #f8fafc !important;">Disclaimer:</strong><br>
                                {disclaimer.get('text', '')} Including the  
                                <a href="{disclaimer.get('links', {}).get('handbook', {}).get('url', '#')}" 
                                target="_blank" 
                                style="color: #93c5fd; text-decoration: underline;">
                                {disclaimer.get('links', {}).get('handbook', {}).get('text', 'Employee Handbook')}
                                </a> and  
                                <a href="{disclaimer.get('links', {}).get('checklist', {}).get('url', '#')}" 
                                target="_blank" 
                                style="color: #93c5fd; text-decoration: underline;">
                                {disclaimer.get('links', {}).get('checklist', {}).get('text', 'Onboarding Orientation Checklist')}
                                </a>.<br>
                                {disclaimer.get('links', {}).get('footer', '')}
                                </div>
                            """)
            # Right Chat Panel 
            with gr.Column(scale=2, elem_classes="right-column"):
                with gr.Column(elem_classes="chat-header"):
                    # Header
                    gr.Markdown(f"## {current_config.domain.upper()} Policy Assistant")                    
                    greeting_display = gr.Markdown(visible=True)
                with gr.Column(elem_classes="chat-scroll-container"):    
                    interface.load(
                        load_greeting,
                        inputs=None,
                        outputs=greeting_display
                    )

                    # Chat components
                    chatbot = gr.Chatbot(
                        elem_id="chatbot",
                        render_markdown=True,
                        avatar_images=(None, "assets/bot.png"),
                        height="100%",
                        show_label=False,
                        type="messages"
                    )
                            
    
                    with gr.Row(elem_classes="textbox-wrapper"):
                        msg = gr.Textbox(
                            placeholder=f"Ask about {current_config.domain} policies...",
                            label="Question",
                            show_label=False,
                            elem_id="question-input",
                            scale=9,
                            container=False
                        )
                        submit_btn = gr.Button(
                            value="➤",  # Submit icon
                            elem_classes="submit-icon",
                            variant="secondary",
                            scale=1,
                            min_width=10
                        )
                    
                    # Event handlers
                    async def respond_and_clear(message: str, chat_history: List[Dict[str, str]]):
                        try:
                            # Add user message
                            chat_history.append({"role": "user", "content": message})
                            yield chat_history, ""
                            
                            # Add assistant placeholder
                            chat_history.append({"role": "assistant", "content": "▌"})
                            yield chat_history, ""
                            
                            # Get response with full chat history context
                            response = generate_response(message, chat_history)
                            answer = format_response(response) if isinstance(response, dict) else str(response)
                            
                            # Stream response
                            clean_text = re.sub(r'<[^>]+>', '', answer)
                            for i in range(0, len(clean_text), CHUNK_SIZE):
                                chat_history[-1]["content"] = clean_text[:i+CHUNK_SIZE]
                                yield chat_history, ""
                                await asyncio.sleep(TYPING_SPEED * CHUNK_SIZE)
                                
                            # Apply final formatting
                            chat_history[-1]["content"] = answer
                            yield chat_history, ""
                            
                        except Exception as e:
                            chat_history[-1]["content"] = "Sorry, I encountered an error"
                            yield chat_history, ""
                    
                    msg.submit(
                        fn=respond_and_clear,
                        inputs=[msg, chatbot],
                        outputs=[chatbot, msg],
                        queue=True
                    ).then(
                        lambda: "",  # Clear message
                        None,
                        msg
                    )

                    submit_btn.click(
                        fn=respond_and_clear,
                        inputs=[msg, chatbot],
                        outputs=[chatbot, msg],
                        queue=True
                    ).then(
                        lambda: "",  # Clear message
                        None,
                        msg
                    )
                    # Add JavaScript via HTML component
                    pdf_opener_js = """
                    <script>
                    document.addEventListener('DOMContentLoaded', function() {
                        document.addEventListener('click', function(e) {
                            if (e.target.classList.contains('page-link')) {
                                e.preventDefault();
                                const href = e.target.getAttribute('href');
                                if (href.includes('/api/v1/pdf/open')) {
                                    window.open(href, '_blank');
                                }
                            }
                        });
                    });
                    </script>
                    """
                    gr.HTML(pdf_opener_js)
            with gr.Column(visible=False) as test_col:
                test_btn = gr.Button("Test Notifications")
                test_btn.click(
                    fn=lambda: notifier.send_pushover("TEST from button") or notifier.send_slack("TEST from button"),
                    outputs=[]
                )
        return interface

if __name__ == "__main__":
    try:
        print("\n=== STARTING APPLICATION ===")
        if not initialize_application():
            print("❌ Initialization failed")
            sys.exit(1)
            
        demo = create_chat_interface()
        
        # Verify QA chain is accessible
        if hasattr(generate_response, 'qa_chain'):
            print(f"✅ QA chain available")
        else:
            print("❌ QA chain not accessible in generate_response")
            
        demo.launch(server_port=7861, server_name="0.0.0.0", share=False,
    debug=True)
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)