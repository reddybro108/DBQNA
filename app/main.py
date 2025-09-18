from fastapi import FastAPI, UploadFile, File
from typing import Dict
import os, tempfile

from .ingestion import pdf_to_text, simple_clean, chunk_text, docx_to_text
from .embedder import embed_texts, get_embedder
from .vector_store import create_collection, upsert_chunks
from .retriever import retrieve
from .generator import generate_answer

app = FastAPI(title="DBQA")

@app.on_event("startup")
def startup():
    dim = get_embedder().get_sentence_embedding_dimension()
    create_collection(dim)

@app.post("/ingest/")
async def ingest_file(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await file.read()
    tmp.write(content)
    tmp.flush()
    tmp.close()

    if suffix in [".pdf"]:
        text = pdf_to_text(tmp.name)
    elif suffix in [".docx"]:
        text = docx_to_text(tmp.name)
    else:
        text = open(tmp.name, "r", encoding="utf-8", errors="ignore").read()

    text = simple_clean(text)
    chunks = chunk_text(text)
    embeds = embed_texts(chunks)
    meta = [{"source": file.filename, "chunk_id": i} for i in range(len(chunks))]
    upsert_chunks(chunks, embeds, meta)
    return {"status": "ok", "n_chunks": len(chunks)}

@app.post("/ask/")
async def ask(payload: Dict):
    question = payload.get("question")
    if not question:
        return {"error": "provide 'question' in JSON body"}
    contexts = retrieve(question, top_k=5)
    answer = generate_answer(question, contexts)
    return {"answer": answer, "contexts": contexts}
