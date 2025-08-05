from pathlib import Path
from .config import current_config  

def validate_structure():
    required = [
        Path("data/domains"),
        Path("backend/config/profiles")
    ]
    for path in required:
        if not path.exists():
            raise RuntimeError(f"Missing required directory: {path}")