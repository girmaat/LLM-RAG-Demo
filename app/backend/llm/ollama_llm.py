from langchain_ollama import OllamaLLM

def get_ollama_llm(model_name="llama2"):
    """
    Returns an instance of OllamaLLM using the specified model.
    Example: model_name = 'llama2', 'mistral', 'phi', etc.
    """
    return OllamaLLM(model=model_name)
