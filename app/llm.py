# rag_hf/llm.py
import os, json, re
from typing import List, Dict, Any
import requests

HF_TOKEN = os.getenv("HF_TOKEN", "")

GENERATION_MODE = os.getenv("GENERATION_MODE", "hf_inference").lower()  # hf_inference | openai_compat
GENERATION_MODEL_ID = os.getenv("GENERATION_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")  # for OpenAI-compatible endpoints (…/v1)

SYSTEM_PROMPT = """You are a cautious assistant for document Q&A.
Use only the provided context to answer. If the context does not contain the answer, say you don't know.
Always include citations by listing the source title and page/section for each claim.
Prefer quoting short evidence in quotes when helpful.
Return JSON only with keys:
- answer: string
- citations: array of {doc_id, title, page, chunk_id, quote}
Do not include any text outside the JSON (no code fences)."""

def build_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for c in chunks:
        m = c["metadata"]
        header = f"[{m.get('title','')} — page {m.get('page','?')} — {m.get('doc_id','')}/{m.get('chunk_index','?')}]"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)

def mistral_chat_template(system_prompt: str, user_prompt: str) -> str:
    # Mistral Instruct v0.2 chat format
    return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}\n[/INST]"

def _safe_json_parse(text: str) -> Dict[str, Any]:
    # strict first
    try:
        return json.loads(text)
    except Exception:
        pass
    # strip fences
    text2 = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text2)
    except Exception:
        pass
    # extract first JSON object
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"answer": text.strip(), "citations": []}

def _ensure_citations(data: Dict[str, Any], retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    if "citations" not in data or not isinstance(data["citations"], list) or not data["citations"]:
        data["citations"] = []
        for r in retrieved:
            m = r["metadata"]
            data["citations"].append({
                "doc_id": m.get("doc_id"),
                "title": m.get("title"),
                "page": m.get("page"),
                "chunk_id": r.get("id", m.get("chunk_id")),
                "quote": r["text"][:180] + ("..." if len(r["text"])>180 else "")
            })
    return data

def ask_llm(query: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    assert HF_TOKEN, "HF_TOKEN is required for Hugging Face calls"
    context_text = build_context(retrieved)
    user_prompt = f"Question: {query}\n\nContext:\n{context_text}\n\nReturn JSON only."

    if GENERATION_MODE == "openai_compat":
        # Use your own HF Inference Endpoint with OpenAI-compatible /chat/completions
        assert OPENAI_BASE_URL, "Set OPENAI_BASE_URL for OpenAI-compatible HF endpoint"
        url = OPENAI_BASE_URL.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
        payload = {
            "model": GENERATION_MODEL_ID,
            "messages": [
                {"role":"system","content": SYSTEM_PROMPT},
                {"role":"user","content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 512
        }
        r = requests.post(url, headers=headers, json=payload, timeout=600)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]

    else:
        # Serverless HF Inference API (text-generation)
        prompt = mistral_chat_template(SYSTEM_PROMPT, user_prompt)
        url = f"https://api-inference.huggingface.co/models/{GENERATION_MODEL_ID}"
        headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.2,
                "top_p": 0.9,
                "do_sample": False,
                "repetition_penalty": 1.05,
                "return_full_text": False
            },
            "options": {"wait_for_model": True}
        }
        r = requests.post(url, headers=headers, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            content = data[0]["generated_text"]
        elif isinstance(data, str):
            content = data
        else:
            content = data.get("choices", [{"text": ""}])[0].get("text", "")

    out = _safe_json_parse(content)
    return _ensure_citations(out, retrieved)