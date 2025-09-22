# rag_hf/eval_ragas.py
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from app.service import QAService

# Replace with your own small eval set
eval_items = [
    {"question": "What is the warranty period?", "reference": "12 months"},
    {"question": "Who is responsible for approvals?", "reference": "The Approvals Committee"},
]

def run_eval():
    qa = QAService()
    rows = []
    for item in eval_items:
        res = qa.ask(item["question"], debug=False)
        citations = res.get("citations", [])
        contexts = []
        for c in citations:
            if isinstance(c, dict):
                quote = c.get("quote") or ""
                contexts.append(quote)
        rows.append({
            "question": item["question"],
            "answer": res.get("answer",""),
            "contexts": contexts,
            "ground_truth": item["reference"],
        })
    ds = Dataset.from_list(rows)
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_recall])
    print(result)

if __name__ == "__main__":
    run_eval()