# app/generator.py
import os
from typing import List, Dict
from huggingface_hub import InferenceClient

# HF client
HF_TOKEN = os.getenv("HF_TOKEN", "")
if not HF_TOKEN:
    raise RuntimeError("Please set HF_TOKEN in your environment variables")

client = InferenceClient(provider="featherless-ai", api_key=HF_TOKEN)
MODEL_ID = os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")

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
    """Generate answer using HF Mistral via Featherless InferenceClient."""
    ctx_text = build_context_text(contexts)
    prompt = PROMPT_TEMPLATE.format(context=ctx_text, question=question)

    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message["content"].strip()
