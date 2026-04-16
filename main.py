import sys
from ingestion import ingest
from agents import run_agent

if __name__ == "__main__":
    if sys.argv[1] == "ingest":
        ingest()
    elif sys.argv[1] == "query":
        print(run_agent(" ".join(sys.argv[2:])))
