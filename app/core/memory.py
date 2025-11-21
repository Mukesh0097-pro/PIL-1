import numpy as np
import time
from typing import List, Dict, Any
from datetime import datetime


class MemoryLayer:
    """Vector Database & History Manager"""

    def __init__(self, embedding_dim: int):
        self.dim = embedding_dim
        # In-memory vector store (Numpy is sufficient for MVP < 1M vectors)
        self.vectors = np.empty((0, embedding_dim))
        self.documents = []
        self.history = []

    def add(self, text: str, vector: np.ndarray, meta: dict = {}):
        """Store a thought/fact"""
        # Ensure vector shape is correct (1, dim)
        if len(vector.shape) == 1:
            vector = vector.reshape(1, -1)

        self.vectors = np.vstack([self.vectors, vector])
        self.documents.append({"text": text, "meta": meta, "time": time.time()})

    def retrieve(self, query_vector: np.ndarray, top_k: int = 2) -> List[Dict]:
        """Retrieve relevant context"""
        if len(self.vectors) == 0:
            return []

        # Cosine Similarity
        # Flatten query if needed
        if len(query_vector.shape) > 1:
            query_vector = query_vector.flatten()

        norm_vec = np.linalg.norm(self.vectors, axis=1)
        norm_query = np.linalg.norm(query_vector)

        scores = np.dot(self.vectors, query_vector) / (norm_vec * norm_query + 1e-8)

        # Get Top-K indices
        top_k = min(top_k, len(self.vectors))
        indices = np.argsort(scores)[::-1][:top_k]

        return [self.documents[i] for i in indices]

    def add_history(self, role: str, content: str):
        """Chat Log"""
        timestamp = datetime.now().strftime("%H:%M")
        self.history.append({"role": role, "content": content, "timestamp": timestamp})
