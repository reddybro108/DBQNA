import os
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from .service import QAService
from .ingest import ingest as run_ingest

app = FastAPI(title="DBQNA - HuggingFace RAG API", version="0.1.0")
qa = QAService()

class AskRequest(BaseModel):
    query: str
    debug: Optional[bool] = False

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest_endpoint(paths: List[str]):
    run_ingest(paths)
    global qa
    qa = QAService()  # reload BM25 state
    return {"ingested": len(paths)}

@app.post("/ask")
async def ask_endpoint(req: AskRequest):
    return qa.ask(req.query, debug=req.debug)

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    saved = []
    base = os.path.join(os.path.dirname(__file__), "uploads")
    os.makedirs(base, exist_ok=True)
    for f in files:
        dest = os.path.join(base, f.filename)
        with open(dest, "wb") as out:
            out.write(await f.read())
        saved.append(dest)
    run_ingest(saved)
    global qa
    qa = QAService()
    return {"uploaded": saved}

# Run: uvicorn rag_hf.api:app --reload