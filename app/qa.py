from app.generator import generate_answer
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Connect to Qdrant
qdrant_client = QdrantClient(host="localhost", port=6333)

# Load the embedding model (outputs 384-dimensional vectors)
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str):
    """Return the embedding vector for the given text using SentenceTransformer."""
    return model.encode(text).tolist()

def retrieve_context(question: str, top_k: int = 3):
    """Retrieve top-k similar chunks from Qdrant"""
    # Query vector via embeddings
    embedding = get_embedding(question)  
    hits = qdrant_client.search(
        collection_name="documents",
        query_vector=embedding,
        limit=top_k
    )
    return [{"text": hit.payload.get("text", "")} for hit in hits]

def answer_question(question: str):
    contexts = retrieve_context(question)
    answer = generate_answer(question, contexts)
    return answer
