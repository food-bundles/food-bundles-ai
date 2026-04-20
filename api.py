from fastapi import FastAPI
from src.agents import run_agent

app = FastAPI()

@app.get("/query")
def query(q: str):
    return {"response": run_agent(q)}
