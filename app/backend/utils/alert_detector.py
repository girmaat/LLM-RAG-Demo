from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.backend.config.alerts import AlertCriteria
import json
import logging
from typing import Dict, Any

class AlertDetector:
    def __init__(self):
        self.logger = logging.getLogger('alert_detector')
        self.logger.setLevel(logging.DEBUG)
        self.criteria = AlertCriteria()
        self.llm = get_llm()
        self.detection_chain = self._build_detection_chain()
        self.logger.info("AlertDetector initialized with detection chain")

    def _build_detection_chain(self):
        prompt = ChatPromptTemplate.from_messages([
                ("system", """Analyze HR policy queries using these criteria:
                SENSITIVE_QUERY: True if about salaries, terminations, or employee data
                URGENT: True if contains 'immediate', 'today', or 'ASAP' 
                ..."""),
            ("human", "QUERY: {query}\nRESPONSE: {response}")
        ])
        return prompt | self.llm | StrOutputParser()

    async def detect(self, query: str, response: str, sources: list) -> Dict[str, bool]:
        DEFAULT_RESULT = {
            "documentation_gap": False,
            "sensitive_query": False,
            "high_value_interaction": False
        }
        
        try:
            from app.backend.llm.llm_factory import get_llm
            # Debug input
            self.logger.debug(f"Detecting alerts for query: {query[:100]}...")
            
            analysis = await self.detection_chain.ainvoke({
                "query": query,
                "response": response
            })
            
            # Force consistent JSON format
            result = json.loads(analysis.strip().lower())
            normalized = {
                "documentation_gap": bool(result.get("documentation_gap", False)),
                "sensitive_query": bool(result.get("sensitive_query", False)),
                "high_value_interaction": bool(result.get("high_value_interaction", False))
            }
            
            self.logger.info(f"Detection completed: {normalized}")
            return normalized
            
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse: {analysis}")
            return DEFAULT_RESULT
        except Exception as e:
            self.logger.error(f"Detection failed: {str(e)}")
            return DEFAULT_RESULT