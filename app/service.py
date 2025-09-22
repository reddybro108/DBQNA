from typing import Dict, Any, List
from .retriever import HybridRetriever, RetrievedChunk, SIM_THRESHOLD
from .llm import ask_llm

class QAService:
    def __init__(self):
        self.retriever = HybridRetriever()

    def ask(self, query: str, debug: bool=False) -> Dict[str, Any]:
        items: List[RetrievedChunk] = self.retriever.retrieve(query)
        if not items:
            payload = {"answer": "I don’t have enough information in the documents to answer that.", "citations": []}
            if debug:
                payload["debug"] = {"retrieved": []}
            return payload

        max_sim = max([it.similarity for it in items])
        if max_sim < SIM_THRESHOLD:
            payload = {"answer": "I don’t have enough information in the documents to answer that.", "citations": []}
            if debug:
                payload["debug"] = {"max_similarity": max_sim, "retrieved": [it.__dict__ for it in items]}
            return payload

        retrieved_for_llm = [{"id": it.id, "text": it.text, "metadata": it.metadata} for it in items]
        result = ask_llm(query, retrieved_for_llm)

        if debug:
            result["debug"] = {"max_similarity": max_sim, "retrieved": [it.__dict__ for it in items]}
        return result