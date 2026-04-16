from fastapi import FastAPI
from agents import run_agent

app = FastAPI()

@app.get("/query")
def query(q: str):
    return {"response": run_agent(q)}
