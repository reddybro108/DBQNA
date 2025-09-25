# DBQNA – Document-Based Question Answering System (Open Source)

This project lets you **upload documents (PDF, DOCX, TXT)** and then ask questions.  
It uses open-source models for embeddings + generation and Qdrant for vector search.

# 📚 DBQNA  

> **DBQNA** is an enterprise-grade Retrieval-Augmented Generation (RAG) system built with FastAPI, OpenAI embeddings, and Pinecone. It ingests documents (PDF/TXT), builds a managed vector database, and exposes APIs for natural language Q&A.

---

## 🗂️ Table of Contents  

- [Overview](#-overview)  
- [Features](#-features)  
- [Architecture](#-architecture)  
- [Tech Stack](#-tech-stack)  
- [Project Structure](#-project-structure)  
- [Getting Started](#-getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
  - [Running Locally](#running-locally)  
  - [Docker](#docker)  
- [Configuration](#-configuration)  
- [API Documentation](#-api-documentation)  
- [Testing](#-testing)  
- [CI/CD](#cicd)  
- [Observability](#-observability)  
- [Security & Compliance](#-security--compliance)  
- [Roadmap](#-roadmap)  
- [Contributing](#-contributing)  
- [License](#-license)  
- [Contact](#-contact)  

---

## 📝 Overview  

**DBQNA** streamlines enterprise Q&A workflows by combining document ingestion, Pinecone vector search, and large language models (OpenAI or HuggingFace).  
It’s built for **production deployments**, with Dockerization, CI/CD, and secure API key management.

---

## ✨ Features  

- 📥 **Document ingestion** (PDF & TXT) into Pinecone vector store  
- 🔍 **RAG-based querying** using OpenAI embeddings  
- ⚡ **FastAPI endpoints** with automatic Swagger UI  
- 🐳 **Docker-ready** for seamless deployment  
- 🔒 **Environment-based configuration** for API keys & models  
- 🧪 **Unit tests & CI/CD** pipeline support  

---

## 🏗 Architecture  

+-------------+ +----------------+ +---------------------+
| Client UI | <-----> | FastAPI App | <-----> | Pinecone Vector DB |
+-------------+ | (app/api.py) | | (Managed Vector Store)|
+----------------+ +---------------------+
|



OpenAI / HuggingFace Models
---

## 🛠 Tech Stack  

| Layer             | Technology                          |
|-------------------|-------------------------------------|
| Backend           | Python 3.11 / FastAPI               |
| Embeddings/LLM    | OpenAI API / HuggingFace Transformers |
| Vector Store      | Pinecone (managed, industry-grade)  |
| Containerization  | Docker / Docker Compose             |
| CI/CD             | GitHub Actions                      |
| Monitoring        | Prometheus / Grafana (optional)     |

---

## 📂 Project Structure  

