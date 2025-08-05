from dataclasses import dataclass
from typing import List

@dataclass
class AlertCriteria:
    """Criteria for determining when to send alerts"""
    sensitive_keywords: List[str] = None
    confidence_threshold: float = 0.85
    
    def __post_init__(self):
        self.sensitive_keywords = self.sensitive_keywords or [
            "confidential",
            "secret",
            "password",
            "alert"
            "client data"
            "security check",        
            "employee record",
            "compensation",
            "salary",
            "termination",
            "disciplinary action",
            "SSN",
            "social security",
            "performance review",
            "payroll"
        ]