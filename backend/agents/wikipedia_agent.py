from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from backend.tools.wikipedia_tool import get_wikipedia_tool
from backend.llm.ollama_llm import get_ollama_llm

def get_wikipedia_agent():
    """
    Creates a simple LangChain agent that can use Wikipedia to answer questions.
    Returns: agent_executor
    """
    # 1. Load the Wikipedia tool
    wikipedia_tool = get_wikipedia_tool()

    # 2. Wrap the tool in LangChain's Tool format
    tool = Tool(
        name="Wikipedia",
        func=wikipedia_tool.run,
        description="Useful for answering questions about general knowledge or definitions from Wikipedia."
    )

    # 3. Load your local LLM (Ollama - llama2)
    llm = get_ollama_llm(model_name="llama2")

    # 4. Create the agent that uses the tool
    agent_executor = initialize_agent(
        tools=[tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent_executor


def query_wikipedia_agent(user_query: str) -> str:
    """
    Takes a user question as input and returns an answer using the Wikipedia agent.
    Example:
        query_wikipedia_agent("What is Zero Trust?")
    """
    agent = get_wikipedia_agent()
    response = agent.run(user_query)
    return response
