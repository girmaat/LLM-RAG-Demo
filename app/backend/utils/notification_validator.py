import os
import requests
from pydantic import BaseModel, FieldValidationInfo, field_validator, HttpUrl
from typing import Optional
from app.backend.config.config import current_config

class NotificationConfig(BaseModel):
    pushover_api_key: Optional[str] = None
    pushover_user_key: Optional[str] = None
    slack_webhook_url: Optional[HttpUrl] = None
    
    @field_validator('pushover_api_key')
    @classmethod
    def validate_pushover_key(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if len(v) != 30:  # Pushover keys are always 30 chars
            raise ValueError("Invalid key length")
        return v


    @field_validator('pushover_user_key')
    @classmethod
    def validate_pushover_user(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if len(v) != 30:
            raise ValueError("Invalid key length")
        return v

    @field_validator('slack_webhook_url')
    @classmethod
    def validate_slack_webhook(cls, v: Optional[HttpUrl], info: FieldValidationInfo) -> Optional[HttpUrl]:
        if v is None:
            return None
            
        try:
            response = requests.post(
                str(v),
                json={"text": "Configuration test message"},
                timeout=3
            )
            if response.status_code == 200:
                return v
            raise ValueError("Slack webhook validation failed")
        except Exception as e:
            raise ValueError(f"Slack webhook validation failed: {str(e)}")

def validate_notification_config():
    """Validate notification configurations if they exist"""
    try:
        config = {
            'pushover_api_key': os.getenv("PUSHOVER_API_KEY"),
            'pushover_user_key': os.getenv("PUSHOVER_USER_KEY"),
            'slack_webhook_url': os.getenv("SLACK_WEBHOOK_URL")
        
        }
        
        print("\n=== NOTIFICATION CONFIG ===")  # Add this
        print(f"Pushover API Key: {'*****' if config['pushover_api_key'] else 'MISSING'}")
        print(f"Pushover User Key: {'*****' if config['pushover_user_key'] else 'MISSING'}")
        print(f"Slack Webhook: {'*****' if config['slack_webhook_url'] else 'MISSING'}")
        
        # Only validate if at least one notification service is configured
        if any(config.values()):
            NotificationConfig(**config)
            
        if not current_config.domain:
            raise ValueError("Notification system requires domain configuration")
            
        return True
    except Exception as e:
        raise ValueError(f"Notification configuration invalid: {str(e)}")