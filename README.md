# DBQNA – Document-Based Question Answering System by using RAG

DBQNA is a production-ready **Retrieval-Augmented Generation (RAG)** framework engineered for high-performance, document-centric question answering. It ingests PDF, DOCX, and TXT files, performs semantic chunking and embedding generation using Hugging Face transformers, persists vector representations in **Qdrant**, and exposes a scalable **FastAPI** service layer for real-time querying.  
The system is container-orchestrated with Docker/Docker Compose and designed with CI/CD pipelines and secure environment configuration to accelerate deployment in enterprise settings.

---

## Table of Contents  
- Features  
- System Architecture  
- Technology Stack  
- Installation  
- Usage  
- API Endpoints  
- Configuration  
- Project Layout  
- Contributing  
- License  

---

## Key Features  

- **Automated Document Ingestion:** Extracts text from PDF/DOCX/TXT, segments it into semantically coherent blocks, generates high-dimensional embeddings, and stores them in Qdrant for millisecond-level retrieval.  
- **End-to-End RAG Pipeline:** Transforms user queries into vector space, retrieves the most relevant chunks, and synthesizes context-aware answers using transformer-based LLMs.  
- **FastAPI Microservice Layer:** Fully asynchronous REST endpoints for ingestion, querying, and vector-store administration, bundled with interactive OpenAPI/Swagger documentation.  
- **Containerized Deployment:** Docker/Docker Compose enable reproducible, cloud-agnostic deployments and local testing environments.  
- **Security & Config Management:** Environment-variable-driven secrets management for API keys, model parameters, and infrastructure URLs.  
- **Automated QA & CI/CD:** Pytest test harness and GitHub Actions integration for continuous validation and deployment.  
- **Observability Ready (Planned):** Hooks for Prometheus/Grafana to monitor latency, query volume, and resource utilization.  

---

## System Architecture  

DBQNA implements a modular, high-throughput RAG pipeline:

1. **Ingestion Layer:** Documents parsed using PyPDF2 (PDF), python-docx (DOCX), or raw text. Semantic chunking (e.g., 512-token blocks) + Hugging Face sentence-transformers for embedding generation.  
2. **Vector Storage Layer:** Qdrant provides high-performance approximate-nearest-neighbor search for embeddings.  
3. **Query Layer:** User questions embedded with the same model, similarity-matched in Qdrant, then context passed to an LLM (Hugging Face or OpenAI) for answer synthesis.  
4. **API Layer:** FastAPI delivers stateless, async REST endpoints for ingestion, querying, and admin tasks.  

```

+-------------+      +----------------+      +-----------------+
|  Client UI  | <--> | FastAPI Server  | <--> | Qdrant Vector DB |
+-------------+      +----------------+      +-----------------+

````

---

## Technology Stack  

| Layer                | Technology                                                      |
|----------------------|------------------------------------------------------------------|
| Backend Framework    | Python 3.11 + FastAPI (asynchronous REST services)               |
| Embeddings / LLM     | Hugging Face Transformers (e.g., all-MiniLM-L6-v2, GPT-2) + optional OpenAI API |
| Vector Store         | Qdrant (local or cloud instance)                                 |
| Document Parsing     | PyPDF2, python-docx, plain text                                  |
| Containerization     | Docker / Docker Compose                                          |
| CI/CD                | GitHub Actions                                                  |
| Testing Framework    | pytest                                                          |
| Monitoring (planned) | Prometheus + Grafana                                            |

---

## Installation  

### Prerequisites  
- Python ≥ 3.11  
- Docker (for Qdrant or full containerized setup)  
- Hugging Face account for model downloads  
- Optional: OpenAI API key  

### Setup Steps  

**Clone the repository:**  
```bash
git clone https://github.com/reddybro108/DBQNA.git
cd DBQNA
````

**Install Python dependencies:**

```bash
pip install -r requirements.txt
```

**Start Qdrant locally (Docker):**

```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Or configure Qdrant Cloud:**
Sign up at [cloud.qdrant.io](https://cloud.qdrant.io), retrieve URL + API key, and update your `.env`.

**Run the FastAPI server locally:**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Usage

**Ingest a document:**

```bash
curl -X POST "http://localhost:8000/ingest" -F "file=@/path/to/document.pdf"
```

**Query your knowledge base:**

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize the document"}'
```

Use the Swagger UI at `http://localhost:8000/docs` for interactive API testing.

---

## API Endpoints

| Endpoint   | Method | Description                     |
| ---------- | ------ | ------------------------------- |
| `/ingest`  | POST   | Upload and index a document     |
| `/query`   | POST   | Submit a natural-language query |
| `/health`  | GET    | API health check                |
| `/vectors` | GET    | Inspect stored vectors          |

---

## Configuration

Add a `.env` file in the project root:

```
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_key
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_LLM_MODEL=gpt2
OPENAI_API_KEY=sk-...   # optional
FASTAPI_PORT=8000
```

`python-dotenv` automatically loads these variables at runtime.

---

## Project Layout

```
DBQNA/
├── app/
│   ├── main.py          # FastAPI entry point
│   ├── ingestion.py     # Document parsing + embedding logic
│   ├── query.py         # Query + RAG orchestration
│   ├── config.py        # Env var handling
├── tests/
│   ├── test_ingestion.py
│   ├── test_query.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Contributing

Contributions are welcome!

* Fork this repo, create a feature branch, commit your changes, and open a Pull Request.
* File bug reports or feature requests via GitHub Issues.
* Follow PEP-8 and include unit tests (`pytest`) for new code paths.

Run tests locally:

```bash
pytest tests/
```

---

## License

This project is licensed under the MIT License – see [LICENSE](LICENSE) for details.

```

This version is ready to paste into your GitHub repository as `README.md`. It makes the tech stack explicit, uses enterprise-style phrasing, and mirrors your actual codebase.
```
