import os
from pathlib import Path
from typing import Dict, Any

class _AppConfig:
    """Protected configuration class"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            object.__setattr__(cls._instance, '_initialized', False)
        return cls._instance
    
    def _initialize(self):
        if getattr(self, '_initialized', False):
            return
            
        # Initialize all attributes using object.__setattr__
        object.__setattr__(self, '_domain', "hr")
        object.__setattr__(self, '_data', {})
        object.__setattr__(self, '_personal', {
            "name": "Girma Debella",
            "title": "Sr. Lead Developer",
            "description": "Driven by a Passion for Applied, Impactful AI",
            "resume_button_text": "Download Resume",
            "blog_url": "https://yourblog.com",
            "profile_image": str(Path("me/profile.png").absolute())
        })
        object.__setattr__(self, '_ui_config', {
            "titles": {
                "assistant_title": "Driven by Curiosity, Fueled by Innovation",
                "start_chatting_title": "See the Product in Action",
                "blog_link_text": "Visit my github porfolio"
            },
            "descriptions": {
                "assistant_description": "Grounded in a diverse IT background—from infrastructure to software engineering—the shift into AI is driven by a deep passion for innovation and intelligent systems. <br /><br />Current focus includes LLMs, RAG, Agentic AI, and tool-augmented reasoning—building adaptive solutions that bridge structured systems with real-time, human-aligned intelligence."
            },
            "chat": {
                "start_chatting_text": "Curious how it all works? Start a chat and experience the AI-powered system—built to showcase real-time reasoning, tools, and intelligent interaction."
            },
            "disclaimer": {
                "text": "This website references content from publicly available documents issued by the Kentucky Personnel Cabinet.",
                "links": {
                    "handbook": {
                        "text": "Employee Handbook",
                        "url": "https://extranet.personnel.ky.gov/DHRA/EmployeeHandbook.pdf" 
                    },
                    "checklist": {
                        "text": "Onboarding Orientation Checklist",
                        "url": "https://extranet.personnel.ky.gov/DHRA/OnboardingOrientationChecklist-AgyGuidance.pdf"
                    },
                    "footer": "This site is not affiliated with or endorsed by the Commonwealth of Kentucky."
                }
            }

        })
        object.__setattr__(self, '_notification_config', {
            "pushover_api_key": os.getenv("PUSHOVER_API_KEY"),
            "pushover_user_key": os.getenv("PUSHOVER_USER_KEY"),
            "slack_webhook_url": os.getenv("SLACK_WEBHOOK_URL")
        })
        object.__setattr__(self, '_initialized', True)

    @property
    def notification_config(self):
        """Read-only access to notification config"""
        return self._notification_config.copy()

    @property
    def domain(self):
        return self._domain
        
    @domain.setter
    def domain(self, value: str):
        object.__setattr__(self, '_domain', value)
        
    @property
    def personal(self) -> Dict[str, Any]:
        return self._personal.copy()
        
    @property
    def ui_config(self) -> Dict[str, Any]:
        return self._ui_config.copy()

    def __setattr__(self, name, value):
        raise AttributeError(
            f"Can't set attribute '{name}' directly - use proper setters or _initialize()"
        )

def get_config():
    """Safe accessor for configuration"""
    instance = _AppConfig()
    if not getattr(instance, '_initialized', False):
        instance._initialize()
    return instance

current_config = get_config()