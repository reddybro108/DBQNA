import os, uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
client = QdrantClient(url=QDRANT_URL)
COLLECTION = "documents"

def create_collection(dim: int):
    cfg = VectorParams(size=dim, distance=Distance.COSINE)
    try:
        client.recreate_collection(collection_name=COLLECTION, vectors_config=cfg)
    except Exception:
        client.create_collection(collection_name=COLLECTION, vectors_config=cfg)

def upsert_chunks(chunks, embeddings, meta_list):
    points = []
    for text, emb, meta in zip(chunks, embeddings, meta_list):
        points.append(PointStruct(id=str(uuid.uuid4()), vector=emb.tolist(), payload={"text": text, **meta}))
    client.upsert(collection_name=COLLECTION, points=points)

def query_vector(vec, top_k=5):
    hits = client.search(collection_name=COLLECTION, query_vector=vec.tolist(), limit=top_k)
    return hits
