import sys
from src.ingestion import ingest
from src.agents import run_agent

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [ingest | query <question>]")
        sys.exit(1)

    if sys.argv[1] == "ingest":
        ingest()
    elif sys.argv[1] == "query":
        print(run_agent(" ".join(sys.argv[2:])))
    else:
        print(f"Unknown command: {sys.argv[1]}")
