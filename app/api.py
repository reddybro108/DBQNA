# app/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

# LangChain community imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Import our ingest function
from app.ingest import ingest  # works if run as a module

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="DBQNA - ML/NLP/GenAI QA API")

VECTOR_STORE_PATH = "vector_store"
EMBEDDINGS_MODEL = HuggingFaceEmbeddings()

# -----------------------------
# Request schemas
# -----------------------------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

# -----------------------------
# Load FAISS DB
# -----------------------------
def load_vector_store(db_path: str = VECTOR_STORE_PATH):
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"FAISS vector store not found at {db_path}. Run /ingest first.")
    db = FAISS.load_local(str(db_path), EMBEDDINGS_MODEL)
    return db

# -----------------------------
# API Endpoints
# -----------------------------
@app.post("/ingest")
def run_ingest(data_dir: str = "data"):
    """
    Ingest all PDFs and TXT files from a folder into FAISS vector store.
    """
    result = ingest(data_dir=data_dir, db_path=VECTOR_STORE_PATH)
    return result

@app.post("/query")
def query_vector_store(req: QueryRequest):
    """
    Query the FAISS vector store using RetrievalQA.
    """
    try:
        db = load_vector_store()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    retriever = db.as_retriever(search_kwargs={"k": req.top_k})
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),  # Uses OpenAI model, you can swap to local LLM
        chain_type="stuff",
        retriever=retriever,
    )
    answer = qa_chain.run(req.question)
    return {"question": req.question, "answer": answer}

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "ok", "message": "DBQNA API is running."}


# # app/api.py
# from __future__ import annotations
# import os
# from pathlib import Path

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel

# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import HuggingFaceEndpoint
# from langchain.chains import RetrievalQA

# # Use package-relative import (requires app/__init__.py)
# from app.ingest import ingest, build_embeddings

# # -----------------------------
# # Config
# # -----------------------------
# VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store")

# HF_TOKEN = os.getenv("HF_TOKEN", "")
# HF_MODEL_ID = os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")

# if not HF_TOKEN:
#     raise RuntimeError("HF_TOKEN is not set. Create a HF token at https://huggingface.co/settings/tokens")

# # Embeddings must match ingest-time settings
# EMBEDDINGS_MODEL = build_embeddings()

# # HF LLM (no OpenAI key needed)
# LLM = HuggingFaceEndpoint(
#     repo_id=HF_MODEL_ID,
#     huggingfacehub_api_token=HF_TOKEN,
#     task="text-generation",
#     max_new_tokens=512,
#     temperature=0.2,
#     model_kwargs={"return_full_text": False},
# )

# # -----------------------------
# # FastAPI app
# # -----------------------------
# app = FastAPI(title="DBQNA - ML/NLP/GenAI QA API")

# # -----------------------------
# # Schemas
# # -----------------------------
# class QueryRequest(BaseModel):
#     question: str
#     top_k: int = 5

# # -----------------------------
# # Helpers
# # -----------------------------
# def load_vector_store(db_path: str = VECTOR_STORE_PATH):
#     p = Path(db_path)
#     if not p.exists():
#         raise FileNotFoundError(f"FAISS vector store not found at {p}. Run /ingest first.")
#     # allow_dangerous_deserialization helps with newer langchain versions
#     db = FAISS.load_local(str(p), EMBEDDINGS_MODEL, allow_dangerous_deserialization=True)
#     return db

# # -----------------------------
# # Endpoints
# # -----------------------------
# @app.post("/ingest")
# def run_ingest(data_dir: str = "data"):
#     """
#     Ingest PDFs/TXT/MD into FAISS vector store.
#     """
#     result = ingest(data_dir=data_dir, db_path=VECTOR_STORE_PATH)
#     return result

# @app.post("/query")
# def query_vector_store(req: QueryRequest):
#     """
#     Query the FAISS vector store using RetrievalQA + HF LLM.
#     """
#     try:
#         db = load_vector_store()
#     except FileNotFoundError as e:
#         raise HTTPException(status_code=400, detail=str(e))

#     retriever = db.as_retriever(search_kwargs={"k": req.top_k})
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=LLM,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=False,  # set True to return docs
#     )
#     answer = qa_chain.run(req.question)
#     return {"question": req.question, "answer": answer}

# @app.get("/")
# def health_check():
#     return {"status": "ok", "message": "DBQNA API is running."}

# # app/api.py
# from __future__ import annotations
# import os
# from pathlib import Path

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel

# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import HuggingFaceEndpoint
# from langchain.chains import RetrievalQA

# # Import from the same package (requires app/__init__.py)
# from .ingest import ingest, build_embeddings  # uses your updated ingest config

# # -----------------------------
# # Config
# # -----------------------------
# VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store")

# HF_TOKEN = os.getenv("HF_TOKEN", "")
# HF_MODEL_ID = os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")

# # Build the same embeddings used at ingest-time
# EMBEDDINGS_MODEL = build_embeddings()

# # Build HF LLM (no OpenAI key needed)
# if not HF_TOKEN:
#     raise RuntimeError("HF_TOKEN is not set. Get a token at https://huggingface.co/settings/tokens")

# LLM = HuggingFaceEndpoint(
#     repo_id=HF_MODEL_ID,
#     huggingfacehub_api_token=HF_TOKEN,
#     task="text-generation",
#     max_new_tokens=512,
#     temperature=0.2,
#     model_kwargs={"return_full_text": False},
# )

# # -----------------------------
# # FastAPI app
# # -----------------------------
# app = FastAPI(title="DBQNA - ML/NLP/GenAI QA API")

# # -----------------------------
# # Schemas
# # -----------------------------
# class QueryRequest(BaseModel):
#     question: str
#     top_k: int = 5

# # -----------------------------
# # Helpers
# # -----------------------------
# def load_vector_store(db_path: str = VECTOR_STORE_PATH):
#     p = Path(db_path)
#     if not p.exists():
#         raise FileNotFoundError(f"FAISS vector store not found at {p}. Run /ingest first.")
#     # allow_dangerous_deserialization is needed on newer langchain-community versions
#     db = FAISS.load_local(str(p), EMBEDDINGS_MODEL, allow_dangerous_deserialization=True)
#     return db

# # -----------------------------
# # Endpoints
# # -----------------------------
# @app.post("/ingest")
# def run_ingest(data_dir: str = "data"):
#     """
#     Ingest PDFs/TXT/MD into FAISS vector store.
#     """
#     result = ingest(data_dir=data_dir, db_path=VECTOR_STORE_PATH)
#     return result

# @app.post("/query")
# def query_vector_store(req: QueryRequest):
#     """
#     Query the FAISS vector store using RetrievalQA + HF LLM.
#     """
#     try:
#         db = load_vector_store()
#     except FileNotFoundError as e:
#         raise HTTPException(status_code=400, detail=str(e))

#     retriever = db.as_retriever(search_kwargs={"k": req.top_k})
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=LLM,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=False,
#     )
#     answer = qa_chain.run(req.question)
#     return {"question": req.question, "answer": answer}

# @app.get("/")
# def health_check():
#     return {"status": "ok", "message": "DBQNA API is running."}

# # app/api.py
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
# from app.ingest import ingest  # Updated ingest.py

# from pathlib import Path

# # -----------------------------
# # FastAPI app
# # -----------------------------
# app = FastAPI(title="DBQNA - ML/NLP/GenAI QA API")

# VECTOR_STORE_PATH = "vector_store"
# EMBEDDINGS_MODEL = HuggingFaceEmbeddings()

# # -----------------------------
# # Request schemas
# # -----------------------------
# class QueryRequest(BaseModel):
#     question: str
#     top_k: int = 5

# # -----------------------------
# # Load FAISS DB
# # -----------------------------
# def load_vector_store(db_path: str = VECTOR_STORE_PATH):
#     db_path = Path(db_path)
#     if not db_path.exists():
#         raise FileNotFoundError(f"FAISS vector store not found at {db_path}. Run /ingest first.")
#     db = FAISS.load_local(str(db_path), EMBEDDINGS_MODEL)
#     return db

# # -----------------------------
# # API Endpoints
# # -----------------------------
# @app.post("/ingest")
# def run_ingest(data_dir: str = "data"):
#     """
#     Ingest all PDFs and TXT files from a folder into FAISS vector store.
#     """
#     result = ingest(data_dir=data_dir, db_path=VECTOR_STORE_PATH)
#     return result

# @app.post("/query")
# def query_vector_store(req: QueryRequest):
#     """
#     Query the FAISS vector store using RetrievalQA.
#     """
#     try:
#         db = load_vector_store()
#     except FileNotFoundError as e:
#         raise HTTPException(status_code=400, detail=str(e))

#     retriever = db.as_retriever(search_kwargs={"k": req.top_k})
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=ChatOpenAI(temperature=0),  # default LLM
#         chain_type="stuff",
#         retriever=retriever,
#     )
#     answer = qa_chain.run(req.question)
#     return {"question": req.question, "answer": answer}

# # -----------------------------
# # Health check
# # -----------------------------
# @app.get("/")
# def health_check():
#     return {"status": "ok", "message": "DBQNA API is running."}









# import os
# from typing import List, Optional
# from fastapi import FastAPI, UploadFile, File
# from pydantic import BaseModel
# from .service import QAService
# from .ingest import ingest as run_ingest

# app = FastAPI(title="DBQNA - HuggingFace RAG API", version="0.1.0")
# qa = QAService()

# class AskRequest(BaseModel):
#     query: str
#     debug: Optional[bool] = False

# @app.get("/health")
# def health():
#     return {"status": "ok"}

# @app.post("/ingest")
# async def ingest_endpoint(paths: List[str]):
#     run_ingest(paths)
#     global qa
#     qa = QAService()  # reload BM25 state
#     return {"ingested": len(paths)}

# @app.post("/ask")
# async def ask_endpoint(req: AskRequest):
#     return qa.ask(req.query, debug=req.debug)

# @app.post("/upload")
# async def upload(files: List[UploadFile] = File(...)):
#     saved = []
#     base = os.path.join(os.path.dirname(__file__), "uploads")
#     os.makedirs(base, exist_ok=True)
#     for f in files:
#         dest = os.path.join(base, f.filename)
#         with open(dest, "wb") as out:
#             out.write(await f.read())
#         saved.append(dest)
#     run_ingest(saved)
#     global qa
#     qa = QAService()
#     return {"uploaded": saved}

# # Run: uvicorn rag_hf.api:app --reload