from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

def get_wikipedia_tool():
    """
    Returns a Wikipedia tool usable in LangChain agents or direct queries.
    """
    wiki_api = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=3000)
    return WikipediaQueryRun(api_wrapper=wiki_api)
