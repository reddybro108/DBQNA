# app/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Our ingest function
from app.ingest import ingest

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="DBQNA - Local QA API")

VECTOR_STORE_PATH = "vector_store"

# -----------------------------
# Embeddings
# -----------------------------
EMBEDDINGS_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# Local HuggingFace LLM
# -----------------------------
model_id = "google/flan-t5-small"  # small model for testing
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
local_llm = HuggingFacePipeline(pipeline=pipe)

# -----------------------------
# Schemas
# -----------------------------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

# -----------------------------
# Load FAISS DB
# -----------------------------
def load_vector_store(db_path: str = VECTOR_STORE_PATH):
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"FAISS vector store not found at {db_path}. Run /ingest first.")
    db = FAISS.load_local(
        str(db_path),
        EMBEDDINGS_MODEL,
        allow_dangerous_deserialization=True
    )
    return db

# -----------------------------
# Endpoints
# -----------------------------
@app.post("/ingest")
def run_ingest(data_dir: str = "data"):
    result = ingest(data_dir=data_dir, db_path=VECTOR_STORE_PATH)
    return result

@app.post("/query")
def query_vector_store(req: QueryRequest):
    try:
        db = load_vector_store()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    retriever = db.as_retriever(search_kwargs={"k": req.top_k})
    qa_chain = RetrievalQA.from_chain_type(
        llm=local_llm,  # Local LLM
        chain_type="stuff",
        retriever=retriever,
    )
    answer = qa_chain.run(req.question)
    return {"question": req.question, "answer": answer}

@app.get("/")
def health_check():
    return {"status": "ok", "message": "DBQNA API is running."}
