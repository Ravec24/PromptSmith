import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def add(a:int , b:int) -> int:
    """add two numbers."""
    return a + b

def multiply(a:int , b:int) -> int:
    """multiply two numbers."""
    return a * b

def subtract(a:int , b:int) -> int:
    """subtract two numbers."""
    return a - b

class ResponseFormat(BaseModel):
    """Response format for the agent."""
    result: str

agent = create_react_agent(
    model,
    tools=[add, subtract, multiply],
    # pre_model_hook=pre_model_hook,
    # post_model_hook=post_model_hook,
    response_format=ResponseFormat,
)

def basic_agent(query: str) -> str:
    resp = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return  resp['structured_response'].result