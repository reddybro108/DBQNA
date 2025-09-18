from .embedder import embed_texts
from .vector_store import query_vector

def retrieve(query, top_k=5):
    q_emb = embed_texts([query])[0]
    hits = query_vector(q_emb, top_k=top_k)
    results = []
    for h in hits:
        results.append({
            "text": h.payload.get("text"),
            "score": h.score,
            "id": h.id
        })
    return results
