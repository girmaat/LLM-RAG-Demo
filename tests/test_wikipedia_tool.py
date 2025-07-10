from backend.tools.wikipedia_tool import get_wikipedia_tool
tool = get_wikipedia_tool()
print(tool.run("What is a performance improvement plan?"))