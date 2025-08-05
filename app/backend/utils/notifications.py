import os
import logging
from typing import Optional
from app.backend.config.config import current_config
from dotenv.main import logger
import requests


class NotificationService:
    def __init__(self):
        # Initialize logger first
        self.logger = logging.getLogger('notifications')
        self.logger.setLevel(logging.DEBUG)
        
        # Initialize all attributes
        self._enabled = False
        self.pushover_api_url = "https://api.pushover.net/1/messages.json"
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        self.detector = None
        
        try:
            from app.backend.utils.alert_detector import AlertDetector
            # Validate configuration
            self._validate_config()
            
            # LAZY IMPORT to break circular dependency
            from app.backend.utils.alert_detector import AlertDetector
            self.detector = AlertDetector()
            
            self._enabled = True
            self.logger.info("Notification service successfully initialized")
            
        except Exception as e:
            self._init_error = str(e)
            logger.error(f"Notification init failed: {e}")
            
    def _validate_config(self):
        """Validate required configuration"""
        if not all([
            os.getenv("PUSHOVER_API_KEY"),
            os.getenv("PUSHOVER_USER_KEY"),
            self.slack_webhook_url
        ]):
            raise ValueError("Missing required notification credentials")

    def send_pushover(self, message: str, title: Optional[str] = None):
        
        if "salary" in message.lower() or "termination" in message.lower():
            priority = 1  # High priority for HR-sensitive topics
        else:
            priority = 0
            
        if not self._enabled:
            self.logger.warning("Pushover disabled - not sending message")
            return
            
        try:
            response = requests.post(
                self.pushover_api_url,
                data={
                    "token": os.getenv("PUSHOVER_API_KEY"),
                    "user": os.getenv("PUSHOVER_USER_KEY"),
                    "message": message,
                    "title": title or f"{current_config.domain.upper()} Alert",
                    "priority": priority
                }
            )
            response.raise_for_status()
            self.logger.info(f"Pushover notification sent: {title}")
        except Exception as e:
            self.logger.error(f"Pushover failed: {str(e)}")

    def send_slack(self, message: str):
        if not self._enabled:
            self.logger.warning("Slack disabled - not sending message")
            return
            
        try:
            response = requests.post(
                self.slack_webhook_url,
                json={"text": message}
            )
            response.raise_for_status()
            self.logger.info("Slack notification sent")
        except Exception as e:
            self.logger.error(f"Slack failed: {str(e)}")

# Singleton instance
notifier = NotificationService()