# rag_hf/query.py
import argparse, json
from app.service import QAService

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Your question")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    qa = QAService()
    res = qa.ask(args.query, debug=args.debug)
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()