from backend.agents.wikipedia_agent import get_wikipedia_agent

agent = get_wikipedia_agent()
response = agent.run("What is Zero Trust security?")
print("\nResponse:\n", response)