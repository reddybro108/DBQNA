import fitz  # type: ignore # pymupdf
import re
from typing import List

def pdf_to_text(path: str) -> str:
    doc = fitz.open(path)
    pages = [page.get_text("text") for page in doc]
    return "\n".join(pages)

def docx_to_text(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def simple_clean(text: str) -> str:
    text = re.sub(r'\s+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def chunk_text(text: str, chunk_size:int=500, overlap:int=50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks
