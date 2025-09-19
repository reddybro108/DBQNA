# DBQNA – Document-Based Question Answering System (Open Source)

This project lets you **upload documents (PDF, DOCX, TXT)** and then ask questions.  
It uses open-source models for embeddings + generation and Qdrant for vector search.

## Features

- Upload documents via REST API.
- Automatically chunk + embed text into a vector database.
- Ask natural language questions, retrieve relevant chunks, and generate grounded answers.
- Fully open source – runs locally.

## Tech Stack

- **FastAPI** – Web framework.
- **Sentence Transformers** – Embeddings.
- **Qdrant** – Vector database.
- **Transformers (Flan-T5)** – Open source text generation.
- **Docker & docker-compose** – To run locally.

## Quickstart

### 1. Clone repo

```bash
git clone https://github.com/yourname/DBQNA.git
cd DBQNA
