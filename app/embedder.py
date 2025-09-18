from sentence_transformers import SentenceTransformer # type: ignore
_model = None

def get_embedder(model_name="all-MiniLM-L6-v2"):
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model

def embed_texts(texts):
    model = get_embedder()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings
