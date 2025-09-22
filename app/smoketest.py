# rag_hf/smoketest.py
import os, requests, json
from .llm import ask_llm

HF_TOKEN = os.getenv("HF_TOKEN", "")
GENERATION_MODEL_ID = os.getenv("GENERATION_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
EMBEDDINGS_MODE = os.getenv("EMBEDDINGS_MODE", "inference").lower()
EMBEDDINGS_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL", "")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")

def check_gen():
    print("Checking generation via HF Inference…")
    dummy = [{"id":"x","text":"This document states the warranty is 12 months.","metadata":{"title":"Manual","page":1,"doc_id":"d","chunk_index":0}}]
    out = ask_llm("What is the warranty period?", dummy)
    assert isinstance(out, dict) and "answer" in out, "Generation failed"
    print("Generation OK:", json.dumps(out, ensure_ascii=False))

def check_embeddings():
    print("Checking embeddings…")
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    texts = ["passage: hello world", "query: warranty period"]
    if EMBEDDINGS_MODE == "tei":
        assert EMBEDDINGS_BASE_URL, "Set EMBEDDINGS_BASE_URL for TEI"
        url = EMBEDDINGS_BASE_URL.rstrip("/") + "/embeddings"
        payload = {"input": texts, "model": EMBEDDING_MODEL_NAME}
    else:
        url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL_NAME}"
        payload = {"inputs": texts, "options": {"wait_for_model": True}}
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    print("Embeddings OK (sample keys):", type(data))

if __name__ == "__main__":
    assert HF_TOKEN, "HF_TOKEN is not set"
    print("HF_TOKEN present")
    print("Model:", GENERATION_MODEL_ID)
    check_gen()
    check_embeddings()
    print("Smoketest passed.")