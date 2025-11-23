import numpy as np
import time
import logging
from sentence_transformers import SentenceTransformer
from jinja2 import Template
from app.core.config import settings
from app.core.memory import MemoryLayer
from app.core.tools import BrowserTool

# Setup Professional Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("indxai.engine")


class PILVAEDecoder:
    """The 'Writing' Brain: Gradient-Free Generator."""

    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.weights = {}
        self.is_trained = False

    def train_analytical(self, X_embeddings):
        """Mathematically solves weights (O(N^2))"""
        if len(X_embeddings) < 2:
            return

        try:
            # 1. Normalize
            self.mean = np.mean(X_embeddings, axis=0)
            self.std = np.std(X_embeddings, axis=0) + 1e-6
            X = (X_embeddings - self.mean) / self.std

            # 2. Encoder (SVD/PCA)
            # Using SVD is more numerically stable than np.cov for wide matrices
            u, s, vt = np.linalg.svd(X.T, full_matrices=False)
            W_enc = u[:, : self.latent_dim]

            # 3. Decoder (Pseudoinverse / Ridge Regression)
            Z = X @ W_enc
            lambda_reg = 0.1
            # Formula: W_dec = (Z.T * Z + λI)^-1 * Z.T * X
            Z_inv = np.linalg.inv(Z.T @ Z + lambda_reg * np.eye(self.latent_dim)) @ Z.T
            W_dec = Z_inv @ X

            self.weights = {"encoder": W_enc, "decoder": W_dec}
            self.is_trained = True
            logger.info(
                f"✅ PIL-VAE Retrained Analytically on {len(X_embeddings)} vectors."
            )
        except Exception as e:
            logger.error(f"Math Error: {e}")


class IndxAI_OS:
    """The Dynamic Operating System"""

    def __init__(self):
        logger.info("🚀 Booting indxai OS (Autonomous Mode)...")
        self.memory = MemoryLayer(embedding_dim=settings.EMBEDDING_DIM)
        self.browser = BrowserTool()
        self.pil_vae = PILVAEDecoder(latent_dim=settings.LATENT_DIM)

        try:
            self.transformer = SentenceTransformer(settings.TRANSFORMER_MODEL)
        except Exception as e:
            logger.warning(f"Transformer failed: {e}")
            self.transformer = None

        # Boot: Load existing memory (Persistence) instead of hardcoded seeds
        vecs = self.memory.get_all_vectors()
        if len(vecs) > 0:
            self.pil_vae.train_analytical(vecs)

    def encode(self, text):
        if self.transformer:
            return self.transformer.encode(text)
        return np.random.randn(settings.EMBEDDING_DIM)

    def learn(self, text_blob: str, source: str = "web"):
        """Instant Ingestion"""
        sentences = [s.strip() for s in text_blob.split(".") if len(s) > 20]
        for s in sentences:
            self.memory.add(s, self.encode(s), source)

        # Retrain instantly
        if len(self.memory.vectors) > 0:
            self.pil_vae.train_analytical(self.memory.get_all_vectors())
        return len(sentences)

    def run_query(self, query: str):
        start = time.time()
        query_vec = self.encode(query)

        # 1. Check Internal Memory
        docs = self.memory.retrieve(query_vec, top_k=3)
        best_score = docs[0]["score"] if docs else 0

        # 2. INTENT ANALYSIS (Privacy Guardrail)
        # If query mentions 'my', 'email', 'file', strictly use Memory.
        is_personal = any(
            w in query.lower()
            for w in ["my", "email", "slack", "private", "file", "doc"]
        )

        # 3. Knowledge Gap Analysis
        # If score is low AND it's not personal, we need the web.
        knowledge_gap = best_score < 0.45

        context = ""
        source_label = "MEMORY"

        if is_personal:
            # PRIVATE MODE
            if best_score > 0.35:
                context = " ".join([d["text"] for d in docs])
            else:
                context = "I checked the internal database but found no matching private records. (Run 'real_connectors.py' to ingest data)"

        elif knowledge_gap:
            # PUBLIC SEARCH MODE
            logger.info(f"🧠 Searching Web for: {query}")
            web_result = self.browser.search(query)

            if web_result and "blocked" not in web_result:
                context = web_result
                source_label = "LIVE WEB"
                # AUTO-LEARNING: Save this so we know it next time
                self.learn(web_result, source="auto_web")
            else:
                context = "I tried to search the web but the connection failed. Falling back to what I know."
        else:
            # MEMORY MODE
            context = " ".join([d["text"] for d in docs])

        # 4. Generation
        tmpl_str = """
        [Analysis]: Processing...
        [Source]: {{source_label}}
        [Response]: {{context}}
        """
        response = Template(tmpl_str).render(source_label=source_label, context=context)
        response = response.replace("\n", " ").strip()

        # Log
        self.memory.add_history("user", query)
        self.memory.add_history("ai", response)

        latency = (time.time() - start) * 1000
        return response, latency
