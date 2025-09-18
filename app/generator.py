import os
import importlib

# Import transformers dynamically to avoid static analyzer import errors and provide a clear error if missing.
try:
    transformers = importlib.import_module("transformers")
    AutoTokenizer = transformers.AutoTokenizer
    AutoModelForSeq2SeqLM = transformers.AutoModelForSeq2SeqLM
    pipeline = transformers.pipeline
except Exception as e:
    raise ImportError("The 'transformers' package is required. Install it with 'pip install transformers'.") from e

torch = None
_TORCH_AVAILABLE = False
try:
    # dynamically import torch so static analyzers don't error if it's not installed
    if importlib.util.find_spec("torch") is not None:
        torch = importlib.import_module("torch")
        _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False

# Default model name if not provided via environment
MODEL = os.environ.get("MODEL", "google/flan-t5-small")

# Load tokenizer and model once
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

# Select device: 0 for first CUDA device if available, otherwise -1 for CPU (pipeline convention)
device = 0 if (_TORCH_AVAILABLE and getattr(torch, "cuda", None) is not None and torch.cuda.is_available()) else -1

# Create pipeline once
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

PROMPT = """
You are an assistant. Use ONLY the context excerpts below to answer the question. If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

def generate_answer(question, contexts):
    assembled = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(contexts)])
    prompt = PROMPT.format(context=assembled, question=question)
    out = pipe(prompt, max_new_tokens=200)
    return out[0]['generated_text']
