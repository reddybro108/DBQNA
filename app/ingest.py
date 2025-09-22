# app/ingest.py
from __future__ import annotations

import os
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple

import numpy as np
import fitz  # PyMuPDF
import requests

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Prefer the new package; fall back if not installed
try:
    import importlib

    _mod = importlib.import_module("langchain_huggingface")
    HuggingFaceEmbeddings = getattr(_mod, "HuggingFaceEmbeddings")
except Exception:
    # Fallback to community (works but shows a deprecation warning)
    from langchain_community.embeddings import HuggingFaceEmbeddings


# -----------------------
# Config (env + sensible defaults)
# -----------------------
# Fast local model by default (good quality/speed)
DEFAULT_LOCAL_MODEL = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/paraphrase-MiniLM-L6-v2"
    # Alternatives:
    # "BAAI/bge-small-en-v1.5"  # better RAG quality, still fast
)

# Provider: "local" or "hf_remote"
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local").lower()

# Remote embedding config (HF serverless or TEI)
HF_TOKEN = os.getenv("HF_TOKEN", "")
EMBEDDINGS_MODE = os.getenv("EMBEDDINGS_MODE", "inference").lower()  # "inference" | "tei"
EMBEDDINGS_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL", "")  # required if EMBEDDINGS_MODE="tei"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))

# Reduce tokenizer noise on Windows
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# -----------------------
# Utilities
# -----------------------
def stable_id(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    return " ".join(s.split())


def batched(items: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


# -----------------------
# Embeddings backends
# -----------------------
class HFRemoteEmbeddings:
    """
    Minimal Embeddings-like class that works with LangChain FAISS.
    Uses Hugging Face APIs:
      - mode='tei': Text Embeddings Inference (OpenAI-compatible /v1/embeddings)
      - mode='inference': serverless feature-extraction pipeline
    """

    def __init__(
        self,
        hf_token: str,
        model: str = "BAAI/bge-small-en-v1.5",
        mode: str = "inference",
        base_url: str | None = None,
        normalize: bool = True,
        batch_size: int = 64,
    ):
        self.hf_token = hf_token
        self.model = model
        self.mode = mode
        self.base_url = base_url
        self.normalize = normalize
        self.batch_size = batch_size
        self.headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}

    def _norm(self, v: List[float]) -> List[float]:
        if not self.normalize:
            return v
        x = np.array(v, dtype=np.float32)
        n = np.linalg.norm(x)
        return (x / (n + 1e-12)).tolist()

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        if self.mode == "tei":
            assert self.base_url, "Set EMBEDDINGS_BASE_URL when EMBEDDINGS_MODE='tei'"
            url = self.base_url.rstrip("/") + "/embeddings"
            payload = {"input": texts, "model": self.model}
            r = requests.post(url, headers=self.headers, json=payload, timeout=120)
            r.raise_for_status()
            return [self._norm(d["embedding"]) for d in r.json()["data"]]
        else:
            # Serverless Inference API
            url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model}"
            payload = {"inputs": texts, "options": {"wait_for_model": True}}
            r = requests.post(url, headers=self.headers, json=payload, timeout=300)
            r.raise_for_status()
            out = r.json()
            vecs: List[List[float]] = []
            for v in out:
                # If token-level, average pool
                if isinstance(v, list) and v and isinstance(v[0], list):
                    vecs.append(np.mean(np.array(v, dtype=np.float32), axis=0).tolist())
                else:
                    vecs.append(v)
            return [self._norm(v) for v in vecs]

    # LangChain expects these two methods
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        bs = self.batch_size
        for batch in batched(texts, bs):
            out.extend(self._embed_batch(batch))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]


def build_embeddings() -> Any:
    if EMBED_PROVIDER == "hf_remote":
        if not HF_TOKEN:
            raise RuntimeError("EMBED_PROVIDER=hf_remote but HF_TOKEN is not set.")
        return HFRemoteEmbeddings(
            hf_token=HF_TOKEN,
            model=EMBEDDING_MODEL_NAME,
            mode=EMBEDDINGS_MODE,
            base_url=EMBEDDINGS_BASE_URL or None,
            normalize=True,
            batch_size=EMBED_BATCH,
        )
    # Local default
    return HuggingFaceEmbeddings(
        model_name=DEFAULT_LOCAL_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": EMBED_BATCH},
    )


# -----------------------
# Extractors
# -----------------------
def extract_text_pdf(pdf_path: Path) -> List[Tuple[str, int]]:
    """
    Extract text per page from a PDF using PyMuPDF.
    Returns list of (text, page_number).
    """
    texts: List[Tuple[str, int]] = []
    try:
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc, start=1):
                txt = page.get_text() or ""
                txt = clean_text(txt)
                if txt:
                    texts.append((txt, i))
    except Exception as e:
        print(f"[WARN] Failed to read PDF {pdf_path}: {e}")
    return texts


def extract_text_txt(txt_path: Path) -> List[Tuple[str, int]]:
    """Return a single 'page' for txt files."""
    try:
        txt = txt_path.read_text(encoding="utf-8", errors="ignore")
        txt = clean_text(txt)
        return [(txt, 1)] if txt else []
    except Exception as e:
        print(f"[WARN] Failed to read TXT {txt_path}: {e}")
        return []


def extract_text_md(md_path: Path) -> List[Tuple[str, int]]:
    return extract_text_txt(md_path)


# -----------------------
# Ingest
# -----------------------
def ingest(
    data_dir: str = "data",
    db_path: str = "vector_store",
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> Dict[str, Any]:
    """
    Load PDFs/TXT/MD files, split into chunks, embed, and ingest into FAISS.
    - Faster default model (MiniLM) for local CPU
    - Optional HF remote embeddings (set EMBED_PROVIDER=hf_remote)
    - Stores metadata: source_path, file_name, page, chunk_index
    """
    t0 = time.time()
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"[ingest] Data dir not found: {data_dir}")
        return {"status": "empty"}

    # Collect files
    files: List[Path] = []
    for ext in ("*.pdf", "*.txt", "*.md"):
        files.extend(data_dir.glob(f"**/{ext}"))

    if not files:
        print("[ingest] No documents found.")
        return {"status": "empty"}

    # Splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    total_files = 0
    total_pages = 0

    for fpath in files:
        total_files += 1
        file_type = fpath.suffix.lower()
        if file_type == ".pdf":
            page_texts = extract_text_pdf(fpath)
        elif file_type in (".txt", ".md"):
            page_texts = extract_text_txt(fpath) if file_type == ".txt" else extract_text_md(fpath)
        else:
            page_texts = []

        if not page_texts:
            continue

        for txt, page_num in page_texts:
            total_pages += 1
            # Split per page to keep page metadata
            chunks = splitter.split_text(txt)
            for idx, ch in enumerate(chunks):
                if len(ch.strip()) < 20:
                    continue
                texts.append(ch)
                metadatas.append(
                    {
                        "source_path": str(fpath.resolve()),
                        "file_name": fpath.name,
                        "page": int(page_num),
                        "chunk_index": int(idx),
                        "doc_id": stable_id(str(fpath.resolve())),
                    }
                )

    if not texts:
        print("[ingest] No non-empty chunks produced.")
        return {"status": "empty"}

    print(f"[ingest] Preparing embeddings with provider='{EMBED_PROVIDER}'...")
    embeddings = build_embeddings()

    # If you use BGE models, you can prepend "passage: " here for better quality:
    # embed_texts = [f"passage: {t}" for t in texts] if "bge" in (DEFAULT_LOCAL_MODEL + EMBEDDING_MODEL_NAME).lower() else texts
    embed_texts = texts

    # Build FAISS index
    db = FAISS.from_texts(embed_texts, embeddings, metadatas=metadatas)

    # Save FAISS store
    Path(db_path).mkdir(parents=True, exist_ok=True)
    db.save_local(db_path)

    dt = time.time() - t0
    print(
        f"[ingest] Ingested {len(texts)} chunks from {total_files} files "
        f"({total_pages} 'pages') into '{db_path}' in {dt:.1f}s."
    )
    return {"status": "success", "files": total_files, "pages": total_pages, "chunks": len(texts), "seconds": dt}


if __name__ == "__main__":
    # Simple CLI
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data", help="Folder containing PDFs/TXT/MD")
    ap.add_argument("--db-path", type=str, default="vector_store", help="FAISS DB output directory")
    ap.add_argument("--chunk-size", type=int, default=1500)
    ap.add_argument("--chunk-overlap", type=int, default=200)
    ap.add_argument(
        "--embed-provider",
        type=str,
        choices=["local", "hf_remote"],
        default=EMBED_PROVIDER,
        help="local (fast MiniLM on CPU) or hf_remote (Hugging Face API)",
    )
    ap.add_argument("--embed-model", type=str, default=DEFAULT_LOCAL_MODEL, help="Local model (only if provider=local)")
    ap.add_argument("--hf-token", type=str, default=HF_TOKEN, help="HF token (only if provider=hf_remote)")
    ap.add_argument("--hf-mode", type=str, choices=["inference", "tei"], default=EMBEDDINGS_MODE)
    ap.add_argument("--hf-base-url", type=str, default=EMBEDDINGS_BASE_URL, help="TEI base URL (…/v1)")

    args = ap.parse_args()

    # Allow CLI overrides to update globals for this run
    if args.embed_provider:
        EMBED_PROVIDER = args.embed_provider.lower()
    if EMBED_PROVIDER == "local":
        os.environ["EMBED_MODEL"] = args.embed_model
        # reflect in default local model variable for this process
        DEFAULT_LOCAL_MODEL = args.embed_model
    else:
        HF_TOKEN = args.hf_token or HF_TOKEN
        EMBEDDINGS_MODE = args.hf_mode or EMBEDDINGS_MODE
        EMBEDDINGS_BASE_URL = args.hf_base_url or EMBEDDINGS_BASE_URL

    ingest(args.data_dir, args.db_path, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)


# ## app/ingest.py
# # from pathlib import Path
# # import fitz  # PyMuPDF
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# # from langchain_community.vectorstores import FAISS
# # from langchain.text_splitter import RecursiveCharacterTextSplitter

# # def extract_text_pdf(pdf_path: Path) -> str:
# #     """Extract text from a PDF using PyMuPDF."""
# #     doc = fitz.open(pdf_path)
# #     text_chunks = []
# #     for page in doc:
# #         text = page.get_text()
# #         if text:
# #             text_chunks.append(text)
# #     return "\n".join(text_chunks)

# # def extract_text_txt(txt_path: Path) -> str:
# #     """Read text from a .txt file."""
# #     return txt_path.read_text(encoding="utf-8")

# # def ingest(data_dir="data", db_path="vector_store", chunk_size=1000, chunk_overlap=100):
# #     """
# #     Load PDFs and TXT files, split into chunks, and ingest into FAISS vector store.
# #     """
# #     data_dir = Path(data_dir)
# #     docs = []

# #     # Collect PDFs
# #     for pdf_file in data_dir.glob("**/*.pdf"):
# #         text = extract_text_pdf(pdf_file)
# #         if text.strip():
# #             docs.append(text)

# #     # Collect TXT files
# #     for txt_file in data_dir.glob("**/*.txt"):
# #         text = extract_text_txt(txt_file)
# #         if text.strip():
# #             docs.append(text)

# #     if not docs:
# #         print("No documents found in the provided data directory.")
# #         return {"status": "empty"}

# #     # Split documents into chunks
# #     splitter = RecursiveCharacterTextSplitter(
# #         chunk_size=chunk_size,
# #         chunk_overlap=chunk_overlap
# #     )
# #     # Updated to match current LangChain API
# #     chunks = []
# #     for doc in docs:
# #         chunks.extend(splitter.split_text(doc))

# #     # Create embeddings
# #     embeddings = HuggingFaceEmbeddings()
# #     db = FAISS.from_texts(chunks, embeddings)

# #     # Save FAISS store
# #     db.save_local(db_path)
# #     print(f"Ingested {len(chunks)} chunks from {len(docs)} documents into '{db_path}'.")

# #     return {"status": "success", "docs": len(docs), "chunks": len(chunks)}

# # if __name__ == "__main__":
# #     # Simple CLI
# #     import argparse
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--data-dir", type=str, default="data", help="Folder containing PDFs and TXT files")
# #     ap.add_argument("--db-path", type=str, default="vector_store", help="FAISS DB output path")
# #     args = ap.parse_args()
# #     ingest(args.data_dir, args.db_path)
