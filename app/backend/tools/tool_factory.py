from pathlib import Path
from typing import Dict, Any, List
from app.backend.config.config import current_config
from langchain_core.documents import Document

def debug_print(*args, **kwargs):
    """Debug print function for logging"""
    print(*args, **kwargs)
    
def get_tool(tool_name: str) -> Any:
    """Complete tool factory with error handling"""
    tool_map = {
        "pdf": _get_pdf_tool,
    }
    
    try:
        if tool_name not in tool_map:
            raise ValueError(f"Unknown tool: {tool_name}")
        return tool_map[tool_name]()
    except Exception as e:
        debug_print(f"⚠️ Tool '{tool_name}' failed: {str(e)}")
        return None

def _get_pdf_tool():
    class PDFTool:
        def run(self, query: str) -> Dict[str, Any]:
            try:
                result = self.chain.invoke({"query": query})
                
                if not result or not result.get("result"):
                    return {
                        "answer": "No relevant information found in documents",
                        "sources": []
                    }
                
                # Ensure sources have proper metadata
                sources = []
                for doc in result.get("source_documents", []):
                    if not hasattr(doc, 'metadata'):
                        doc.metadata = {}
                    sources.append(doc)
                
                return {
                    "answer": str(result["result"]),
                    "sources": sources
                }
                
            except Exception as e:
                self.logger.error(f"PDF query failed: {str(e)}")
                return {
                    "answer": "Error searching documents",
                    "sources": []
                }