from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()  

def get_llm():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")
        
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        streaming=True,
        max_tokens=1024
    )