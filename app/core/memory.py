import sqlite3
import numpy as np
import faiss
import time
import threading
from typing import List, Dict, Optional
import structlog

logger = structlog.get_logger(__name__)


class MemoryLayer:
    """
    Persistent Memory with FAISS Vector Index + SQLite Storage.
    Thread-safe with connection pooling.
    """

    def __init__(self, db_path: str = "brain.db", embedding_dim: int = 384):
        self.db_path = db_path
        self.dim = embedding_dim
        self.history: List[Dict] = []  # RAM Cache for UI

        # FAISS index for fast similarity search
        self.index: Optional[faiss.IndexFlatIP] = None
        self._text_lookup: List[str] = []  # Maps FAISS index -> text

        # Thread safety
        self._lock = threading.RLock()
        self._local = threading.local()

        self._init_db()
        self._init_faiss_index()
        self._load_history_from_disk()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self.db_path, check_same_thread=False, timeout=30.0
            )
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
        return self._local.connection

    def _init_db(self):
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                vector BLOB,
                source TEXT,
                timestamp REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                content TEXT,
                timestamp REAL
            )
        """)
        # Add index for faster queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_knowledge_source ON knowledge(source)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_timestamp ON history(timestamp)"
        )
        conn.commit()

    def _init_faiss_index(self):
        """Initialize FAISS index and load existing vectors."""
        with self._lock:
            # Use Inner Product for cosine similarity (vectors should be normalized)
            self.index = faiss.IndexFlatIP(self.dim)

            # Load existing vectors from DB
            conn = self._get_connection()
            try:
                cursor = conn.execute("SELECT text, vector FROM knowledge ORDER BY id")
                vectors = []
                for row in cursor:
                    text, vec_blob = row
                    vec = np.frombuffer(vec_blob, dtype=np.float32)
                    # Normalize for cosine similarity
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    vectors.append(vec)
                    self._text_lookup.append(text)

                if vectors:
                    vectors_array = np.array(vectors, dtype=np.float32)
                    self.index.add(vectors_array)
                    logger.info("faiss_index_loaded", num_vectors=len(vectors))
            except Exception as e:
                logger.error("faiss_init_failed", error=str(e))

    def _load_history_from_disk(self):
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT role, content FROM history ORDER BY id DESC LIMIT 50"
            )
            rows = cursor.fetchall()
            for row in reversed(rows):
                self.history.append({"role": row[0], "content": row[1]})
        except Exception as e:
            logger.warning("history_load_failed", error=str(e))

    def add(self, text: str, vector: np.ndarray, source: str = "auto"):
        """Add a new memory entry with FAISS indexing."""
        with self._lock:
            conn = self._get_connection()
            vec_blob = vector.astype(np.float32).tobytes()
            conn.execute(
                "INSERT INTO knowledge (text, vector, source, timestamp) VALUES (?, ?, ?, ?)",
                (text, vec_blob, source, time.time()),
            )
            conn.commit()

            # Add to FAISS index
            vec_normalized = vector.astype(np.float32)
            norm = np.linalg.norm(vec_normalized)
            if norm > 0:
                vec_normalized = vec_normalized / norm
            self.index.add(vec_normalized.reshape(1, -1))
            self._text_lookup.append(text)

            logger.debug("memory_added", source=source, text_length=len(text))

    def retrieve(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict]:
        """Fast similarity search using FAISS."""
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return []

            # Normalize query vector
            query_vec = query_vector.astype(np.float32)
            norm = np.linalg.norm(query_vec)
            if norm > 0:
                query_vec = query_vec / norm
            query_vec = query_vec.reshape(1, -1)

            # FAISS search
            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(query_vec, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self._text_lookup):
                    results.append(
                        {"text": self._text_lookup[idx], "score": float(score)}
                    )

            return results

    def get_all_vectors(self) -> np.ndarray:
        """
        Get all vectors from FAISS index (for VAE training).
        """
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return np.empty((0, self.dim), dtype=np.float32)

            # Reconstruct vectors from FAISS index
            n_vectors = self.index.ntotal
            vectors = np.zeros((n_vectors, self.dim), dtype=np.float32)
            for i in range(n_vectors):
                vectors[i] = self.index.reconstruct(i)
            return vectors

    def add_history(self, role: str, content: str):
        """Add chat history entry."""
        with self._lock:
            self.history.append({"role": role, "content": content})
            conn = self._get_connection()
            conn.execute(
                "INSERT INTO history (role, content, timestamp) VALUES (?, ?, ?)",
                (role, content, time.time()),
            )
            conn.commit()

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        with self._lock:
            conn = self._get_connection()
            knowledge_count = conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[
                0
            ]
            history_count = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]

            return {
                "knowledge_entries": knowledge_count,
                "history_entries": history_count,
                "faiss_vectors": self.index.ntotal if self.index else 0,
                "cached_history": len(self.history),
            }

    def clear_knowledge(self):
        """Clear all knowledge (for testing/reset)."""
        with self._lock:
            conn = self._get_connection()
            conn.execute("DELETE FROM knowledge")
            conn.commit()

            # Reset FAISS index
            self.index = faiss.IndexFlatIP(self.dim)
            self._text_lookup.clear()
            logger.info("memory_cleared")

    def close(self):
        """Close database connections."""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
