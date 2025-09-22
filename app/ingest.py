# app/ingest.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
import fitz  # PyMuPDF

def extract_text(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    text = []
    doc = fitz.open(str(pdf_path))
    for page in doc:
        page_text = page.get_text("text")
        if page_text:
            text.append(page_text)
    return "\n".join(text)

def ingest(data_dir="data/pdf", db_path="vector_store"):
    """Load PDFs and ingest into FAISS vector DB."""
    docs = []
    pdf_files = list(Path(data_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {data_dir}")
        return {"status": "failed", "docs": 0}

    for pdf in pdf_files:
        text = extract_text(pdf)
        if text.strip():
            docs.append(text)

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_texts(docs, embeddings)
    db.save_local(db_path)
    print(f"Ingested {len(docs)} documents into {db_path}")
    return {"status": "success", "docs": len(docs)}
