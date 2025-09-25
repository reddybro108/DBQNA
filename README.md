DBQNA - Document-Based Question Answering System

DBQNA is an open-source Retrieval-Augmented Generation (RAG) system designed for enterprise-grade document question answering. Upload documents in PDF, DOCX, or TXT formats, and query them using natural language. It leverages HuggingFace models for embeddings and text generation, Qdrant for efficient vector search, and FastAPI for scalable APIs. Built for production, it includes Docker support, secure configuration, and CI/CD integration.
This project is ideal for building knowledge bases, internal search tools, or AI-powered chatbots that answer questions based on uploaded documents.
Table of Contents

Features
Architecture
Tech Stack
Installation
Usage
API Endpoints
Configuration
Project Structure
Contributing
License

Features

📥 Document Ingestion: Upload and process PDF, DOCX, and TXT files. Extracts text, splits into chunks, generates embeddings, and stores them in Qdrant.
🔍 RAG Pipeline: Queries are embedded, matched against document chunks via vector search, and augmented for LLM-based answer generation.
⚡ FastAPI Backend: Exposes RESTful APIs for document ingestion, querying, and vector store management, with Swagger UI for interactive testing.
🐳 Containerization: Docker and Docker Compose for easy local or cloud deployment.
🔒 Secure Configuration: Environment variables for API keys, model selection, and database credentials.
🧪 Testing & CI/CD: Unit tests with pytest and automated workflows via GitHub Actions.
📊 Monitoring (Planned): Support for Prometheus and Grafana integration for query and system metrics.

Architecture
DBQNA follows a modular Retrieval-Augmented Generation (RAG) pipeline:

Ingestion: Documents are parsed (using PyPDF2 for PDFs, python-docx for DOCX, or plain text for TXT), split into chunks (e.g., 512-token segments), and converted to embeddings using HuggingFace's sentence-transformers.
Vector Storage: Embeddings are stored in Qdrant for fast similarity search.
Querying: User queries are embedded, searched against Qdrant for top-k relevant chunks, and passed to an LLM (e.g., HuggingFace's gpt2 or OpenAI's GPT-3.5) for answer generation.
API Layer: FastAPI serves asynchronous endpoints for ingestion and querying.

+-------------+     +----------------+     +---------------------+
| Client UI/  | <--> | FastAPI Server | <--> | Qdrant Vector DB    |
| API Caller  |     | (app/main.py)  |     | (Local/Cloud Vector |
+-------------+     +----------------+     | Store with Embeddings)|
                                    +---------------------+

Tech Stack



Layer
Technology



Backend
Python 3.11 / FastAPI


Embeddings/LLM
HuggingFace Transformers (e.g., all-MiniLM-L6-v2, gpt2) / OpenAI API (optional)


Vector Store
Qdrant (open-source, scalable)


Document Parsing
PyPDF2, python-docx, plain text


Containerization
Docker / Docker Compose


CI/CD
GitHub Actions


Testing
pytest


Monitoring
Prometheus / Grafana (planned)


Installation
Prerequisites

Python: 3.11 or higher
Docker: For containerized setup
HuggingFace: Access to models (free, create an account at huggingface.co)
Qdrant: Local instance or Qdrant Cloud account
Optional: OpenAI API key for proprietary models

Steps

Clone the Repository:
git clone https://github.com/reddybro108/DBQNA.git
cd DBQNA


Install Dependencies:Create a requirements.txt with:
fastapi==0.115.0
uvicorn==0.30.6
qdrant-client==1.11.0
transformers==4.44.2
sentence-transformers==3.0.1
pypdf2==3.0.1
python-docx==1.1.2
python-dotenv==1.0.1
pytest==8.3.2

Then run:
pip install -r requirements.txt


Set Up Qdrant:

Local: Start Qdrant with Docker:docker run -p 6333:6333 qdrant/qdrant


Cloud: Sign up at cloud.qdrant.io and get your API key and URL.


Docker Setup (Optional):Use the provided docker-compose.yml:
docker-compose up -d



Usage
Start the FastAPI Server
Run locally:
uvicorn app.main:app --host 0.0.0.0 --port 8000

Access the API at http://localhost:8000. Use Swagger UI at http://localhost:8000/docs for interactive testing.
Ingest a Document
Upload a document via API:
curl -X POST "http://localhost:8000/ingest" -F "file=@/path/to/document.pdf"

This parses the document, generates embeddings, and stores them in Qdrant.
Query Documents
Ask a question:
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d '{"question": "What is the main topic of the document?"}'

Example Response:
{
  "answer": "The document discusses enterprise-grade question answering.",
  "sources": ["document.pdf:chunk_1"],
  "confidence": 0.95
}

API Endpoints



Endpoint
Method
Description



/ingest
POST
Upload and process a document


/query
POST
Query documents with a question


/health
GET
Check API server status


/vectors
GET
List stored vectors (admin only)


Explore full API docs at /docs.
Configuration
Create a .env file in the project root:
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_key_here  # If using Qdrant Cloud
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_LLM_MODEL=gpt2
OPENAI_API_KEY=sk-...  # Optional, for OpenAI models
FASTAPI_PORT=8000

The app loads these using python-dotenv. Update app/main.py to read these variables:
from dotenv import load_dotenv
import os
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")

Project Structure
DBQNA/
├── app/
│   ├── main.py          # FastAPI app entry point
│   ├── ingestion.py     # Document parsing and embedding logic
│   ├── query.py         # RAG pipeline for querying
│   ├── config.py        # Environment variable handling
├── tests/
│   ├── test_ingestion.py # Unit tests for ingestion
│   ├── test_query.py     # Unit tests for querying
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Multi-container setup
├── requirements.txt     # Python dependencies
├── .env.example         # Sample env file
├── .gitignore           # Git ignore rules
└── README.md            # This file

Note: If you haven’t created these files yet, use this as a blueprint for organizing your code.
Contributing
We welcome contributions to DBQNA! To get started:

Fork the repository.
Create a feature branch: git checkout -b feature/your-feature.
Commit changes: git commit -m "Add your feature".
Push and open a Pull Request.
Report bugs or suggest features via GitHub Issues.

Please follow PEP8 style guidelines and include tests for new features. Run tests with:
pytest tests/

License
This project is licensed under the MIT License - see the LICENSE file for details.