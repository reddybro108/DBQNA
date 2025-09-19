# app/generator.py
import os
import logging
import requests
from typing import List, Dict

logger = logging.getLogger("dbqna.generator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)

# Config via env
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")  # default model
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 256))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.0))

PROMPT_TEMPLATE = """You are a helpful assistant that answers user questions using ONLY the provided context.
Cite the relevant context chunk numbers in square brackets like [1], [2].
If the answer is not fully contained in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

def build_context_text(contexts: List[Dict]) -> str:
    """Assemble contexts with chunk numbers for citation."""
    return "\n\n".join([f"[{i+1}] {c.get('text','')}" for i, c in enumerate(contexts)])

def generate_answer(question: str, contexts: List[Dict]) -> str:
    """Generate answer using Hugging Face Inference API."""
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN environment variable is required for Hugging Face Inference API.")

    ctx_text = build_context_text(contexts)
    prompt = PROMPT_TEMPLATE.format(context=ctx_text, question=question)

    url = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": MAX_TOKENS, "temperature": TEMPERATURE},
        "options": {"wait_for_model": True}
    }

    logger.info("Sending request to Hugging Face Inference API...")
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        logger.error(f"HF Inference API error {response.status_code}: {response.text}")
        raise RuntimeError(f"HF Inference API error {response.status_code}: {response.text}")

    data = response.json()
    # HF may return different shapes
    if isinstance(data, list) and len(data) > 0:
        text = data[0].get("generated_text") or data[0].get("text") or str(data[0])
    elif isinstance(data, dict):
        text = data.get("generated_text") or data.get("text") or str(data)
    else:
        text = str(data)

    return text.strip()
