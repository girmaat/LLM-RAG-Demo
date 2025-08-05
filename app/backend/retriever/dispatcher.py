from typing import Dict, List
from app.backend.config.config import current_config

def debug_print(*args, **kwargs):
    """Debug print function for logging"""
    print(*args, **kwargs)

class ToolDispatcher:
    _DOMAIN_TOOLS = {
        "finance": {
            "primary": "pdf",
            "fallbacks": []  
        },        
        "hr": {
            "primary": "pdf",
            "fallbacks": []  
        }
    }

    @classmethod
    def get_tools(cls, query: str) -> Dict[str, List[str]]:
        """Tool dispatcher for PDF only"""
        domain = current_config.domain
        debug_print(f"🐞 Dispatching tools for domain: {domain}")
        
        print(f"🐞 Current domain: {domain}")
        print(f"🐞 Available domains: {list(cls._DOMAIN_TOOLS.keys())}")
    
        if domain not in cls._DOMAIN_TOOLS:
            print(f"❌ WARNING: No tools configured for domain: {domain}")
            print("🐞 Falling back to default PDF tool")
            return {"primary": "pdf", "fallbacks": []}
        print(f"🐞 Selected tools: {cls._DOMAIN_TOOLS[domain]}")
        return cls._DOMAIN_TOOLS[domain]