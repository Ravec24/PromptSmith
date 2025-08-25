from fastapi import FastAPI
from pydantic import BaseModel
from agent import basic_agent

app = FastAPI()

class RequestData(BaseModel):
    query: str

def ai_agent_response(query: str) -> str:
    resp = basic_agent(query)
    return resp

@app.post("/agent")
async def run_agent(request: RequestData):
    response = ai_agent_response(request.query)
    return {"response": response}