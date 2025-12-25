import time
import json
import hashlib
import numpy as np
from typing import List, Dict, Optional, Generator
from sentence_transformers import SentenceTransformer
from ddgs import DDGS
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import structlog

from app.core.pil_vae import PILVAE
from app.core.gmail_tool import GmailTool
from app.core.config import settings
from app.core.memory import MemoryLayer

# Configure structured logging
logger = structlog.get_logger(__name__)


class IndxAI_OS:
    """
    PIL-VAE Hybrid AI Engine with improved scalability.
    """

    # Class-level cache for web searches
    _search_cache: Dict[str, List[Dict]] = {}
    _cache_max_size: int = 100
    _cache_ttl: float = 300.0  # 5 minutes
    _cache_timestamps: Dict[str, float] = {}

    def __init__(self):
        logger.info("engine_initializing", component="PIL-VAE Hybrid Engine")

        # Load Embedding Model
        self.embedder = SentenceTransformer(settings.TRANSFORMER_MODEL)

        # Initialize PIL-VAE with settings
        self.vae = PILVAE(
            input_dim=settings.EMBEDDING_DIM,
            latent_dim=settings.LATENT_DIM,
            hidden_dim=128,
        )

        # Initialize Tools
        self.gmail = GmailTool()

        # Memory Stores
        self.memory = MemoryLayer(embedding_dim=settings.EMBEDDING_DIM)
        self.memory_texts: List[str] = []
        self.memory_embeddings: Optional[np.ndarray] = None

        # Pre-load memory from DB
        self._hydrate_memory()

        # Compatibility
        self.mode = "assistant"

        # Streaming config
        self.stream_delay: float = 0.0  # No artificial delay by default

        logger.info("engine_ready")

    def _hydrate_memory(self):
        """Load existing vectors from DB into VAE training set"""
        vecs = self.memory.get_all_vectors()
        if len(vecs) > 0:
            self.memory_embeddings = vecs
            self.vae.fit(self.memory_embeddings)
            logger.info("memory_hydrated", num_vectors=len(vecs))

    def _extract_keywords(self, query: str) -> str:
        """Basic keyword extraction for better search queries"""
        stopwords = {
            "my",
            "last",
            "gmail",
            "email",
            "emails",
            "message",
            "messages",
            "from",
            "about",
            "check",
            "search",
            "find",
            "in",
            "the",
            "for",
            "to",
            "on",
            "with",
            "show",
            "me",
            "get",
            "read",
            "fetch",
        }
        words = query.lower().split()
        keywords = [w for w in words if w not in stopwords]
        return " ".join(keywords)

    def _clean_web_query(self, query: str) -> str:
        """Remove conversational prefixes for better web search results"""
        prefixes = [
            "tell me about",
            "what is",
            "who is",
            "search for",
            "find",
            "show me",
        ]
        q_lower = query.lower()
        for p in prefixes:
            if q_lower.startswith(p):
                return query[len(p) :].strip()
        return query

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for search queries"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self._cache_timestamps:
            return False
        return (time.time() - self._cache_timestamps[cache_key]) < self._cache_ttl

    def _clean_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            k
            for k, t in self._cache_timestamps.items()
            if current_time - t > self._cache_ttl
        ]
        for k in expired_keys:
            self._search_cache.pop(k, None)
            self._cache_timestamps.pop(k, None)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _search_with_backend(
        self, query: str, max_results: int, backend: str
    ) -> List[Dict]:
        """Execute search with specific backend and retry logic"""
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results, backend=backend))

    def search_web(self, query: str, max_results: int = 5) -> List[Dict]:
        """Robust web search with caching and fallback."""
        # Check cache first
        cache_key = self._get_cache_key(query)
        if cache_key in self._search_cache and self._is_cache_valid(cache_key):
            logger.debug("search_cache_hit", query=query[:50])
            return self._search_cache[cache_key]

        # Clean expired entries periodically
        if len(self._search_cache) > self._cache_max_size:
            self._clean_cache()

        results = []
        try:
            # Try Lite backend first (faster)
            results = self._search_with_backend(query, max_results, "lite")
            logger.info("search_success", backend="lite", num_results=len(results))
        except Exception as e:
            logger.warning("search_backend_failed", backend="lite", error=str(e))
            try:
                # Fallback to HTML backend
                results = self._search_with_backend(query, max_results, "html")
                logger.info("search_success", backend="html", num_results=len(results))
            except Exception as e2:
                logger.error("search_all_backends_failed", error=str(e2))

        # Cache results
        if results:
            self._search_cache[cache_key] = results
            self._cache_timestamps[cache_key] = time.time()

        return results

    def update_memory(self, new_texts: List[str], source: str = "web"):
        """Embeds new texts and updates the VAE."""
        if not new_texts:
            return

        # 1. Embed Text
        new_embs = self.embedder.encode(new_texts)

        # 2. Update Storage (DB)
        for text, emb in zip(new_texts, new_embs):
            self.memory.add(text, emb, source)

        # 3. Update Local Cache & VAE
        if self.memory_embeddings is None:
            self.memory_embeddings = new_embs
        else:
            self.memory_embeddings = np.vstack([self.memory_embeddings, new_embs])

        # 4. Fit VAE on ALL memory (Closed-form is fast)
        self.vae.fit(self.memory_embeddings)

        logger.info("memory_updated", source=source, num_entries=len(new_texts))

    def learn_new_data(self, text_blob: str):
        """Compatibility wrapper for endpoint."""
        # Split by sentences roughly
        sentences = [s.strip() for s in text_blob.split(".") if len(s.strip()) > 10]
        if sentences:
            self.update_memory(sentences, source="user_training")
            logger.info("training_complete", num_sentences=len(sentences))

    def get_reasoning_response(
        self,
        query_vec: np.ndarray,
        query_text: str,
        forced_context: Optional[List[str]] = None,
    ) -> str:
        """
        V4 Improved Reasoning Logic:
        Latent -> Reconstruct -> k-NN -> Semantic Composition
        """
        # 1. VAE "Reasoning" Step
        z = self.vae.encode(query_vec)
        e_gen = self.vae.decode(z)

        # 2. Nearest-Neighbor Retrieval (Cosine Similarity)
        docs = self.memory.retrieve(e_gen, top_k=5)

        top_texts: List[str] = []
        top_scores: List[float] = []

        # Helper function to check if two texts are semantically similar (duplicates)
        def is_duplicate(
            new_text: str, existing_texts: List[str], threshold: float = 0.7
        ) -> bool:
            """Check if new_text is too similar to any existing text."""
            new_words = set(new_text.lower().split())
            for existing in existing_texts:
                existing_words = set(existing.lower().split())
                if not new_words or not existing_words:
                    continue
                # Jaccard similarity
                intersection = len(new_words & existing_words)
                union = len(new_words | existing_words)
                similarity = intersection / union if union > 0 else 0
                if similarity > threshold:
                    return True
            return False

        # Priority 1: Forced Context (Live Data) - but filter for relevance
        if forced_context:
            # Score forced context by similarity to query
            query_words = set(query_text.lower().split())
            scored_context = []
            for text in forced_context[:5]:
                text_words = set(text.lower().split())
                # Calculate word overlap score
                overlap = len(query_words & text_words)
                relevance = overlap / max(len(query_words), 1)
                scored_context.append((text, 0.8 + 0.2 * relevance))

            # Sort by relevance
            scored_context.sort(key=lambda x: x[1], reverse=True)
            for text, score in scored_context[:3]:
                if text.strip() and not is_duplicate(text, top_texts):
                    top_texts.append(text)
                    top_scores.append(score)

        # Priority 2: Memory Retrieval - with relevance threshold
        if docs:
            # Extract key terms from query for relevance checking
            query_lower = query_text.lower()
            query_terms = set(query_lower.split())

            for d in docs:
                text = d["text"]
                score = d["score"]

                # Skip if already included or empty
                if text in top_texts or not text.strip():
                    continue

                # Apply relevance threshold - skip low-scoring results
                if score < 0.6:
                    continue

                # Additional semantic check: ensure result relates to query
                text_lower = text.lower()
                # Check if any significant query term appears in the result
                significant_terms = [t for t in query_terms if len(t) > 3]
                has_relevance = any(term in text_lower for term in significant_terms)

                # Check for duplicate/similar content
                if is_duplicate(text, top_texts):
                    continue

                # Only include if score is high OR has direct term relevance
                if score >= 0.7 or has_relevance:
                    top_texts.append(text)
                    top_scores.append(score)

        if not top_texts:
            return "I couldn't find any relevant information. Try rephrasing your question."

        # Limit to top 5 total
        top_texts = top_texts[:5]
        top_scores = top_scores[:5]

        # 3. Improved Semantic Analysis
        # Extract meaningful concepts using TF-IDF-like weighting
        from collections import Counter

        # Stopwords for better concept extraction
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "and",
            "but",
            "if",
            "or",
            "because",
            "until",
            "while",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "they",
            "them",
            "their",
            "what",
            "which",
            "who",
            "whom",
            "http",
            "https",
            "www",
            "com",
            "org",
        }

        combined_text = " ".join(top_texts)
        # Clean and tokenize
        words = [
            w.strip(".,!?;:'\"()[]{}")
            for w in combined_text.lower().split()
            if len(w) > 3 and w.lower() not in stopwords
        ]

        # Get most common meaningful words
        word_counts = Counter(words)
        concepts = [word for word, _ in word_counts.most_common(5) if word][:3]

        if not concepts:
            concepts = ["information", "data", "results"]

        # 4. Generate structured response
        response = f"**Analysis**: Key topics identified: {', '.join(concepts)}.\n\n"
        response += "**Findings**:\n"

        for i, (txt, score) in enumerate(zip(top_texts, top_scores), 1):
            # Clean and format display text
            display_txt = txt.replace("\n", " ").strip()

            # Remove ellipsis artifacts from web snippets
            display_txt = display_txt.replace(" ... ", " ").replace("...", "")
            display_txt = " ".join(display_txt.split())  # Normalize whitespace

            # Smart truncation at sentence boundary
            if len(display_txt) > 250:
                last_period = display_txt[:250].rfind(".")
                last_question = display_txt[:250].rfind("?")
                last_exclaim = display_txt[:250].rfind("!")
                best_break = max(last_period, last_question, last_exclaim)

                if best_break > 80:
                    display_txt = display_txt[: best_break + 1]
                else:
                    # Find any sentence boundary, even earlier
                    for punct in [". ", "? ", "! "]:
                        idx = display_txt[:250].rfind(punct)
                        if idx > 50:
                            display_txt = display_txt[: idx + 1]
                            break
                    else:
                        display_txt = display_txt[:250] + "..."

            # Format with confidence score
            response += f"{i}. [{score:.2f}] {display_txt}\n\n"

        # 5. Generate intelligent conclusion
        response += "**Summary**: "

        # Extract the most informative sentence from top result
        if top_texts:
            first_result = top_texts[0].replace("\n", " ").strip()
            # Clean ellipsis artifacts from web snippets
            first_result = first_result.replace(" ... ", " ").replace("...", "")
            first_result = " ".join(first_result.split())  # Normalize whitespace

            # Better sentence splitting - handle common abbreviations
            import re

            # Split on sentence boundaries but not on common abbreviations
            sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"
            sentences = [
                s.strip()
                for s in re.split(sentence_pattern, first_result)
                if len(s.strip()) > 20
            ]

            # If regex didn't work well, fall back to simple split
            if not sentences:
                sentences = [
                    s.strip() for s in first_result.split(".") if len(s.strip()) > 20
                ]

            if sentences:
                # Find the first complete, factual sentence (prefer sentences with dates/facts)
                query_words = set(query_text.lower().split())

                # Score sentences by relevance and completeness
                def score_sentence(s):
                    s_lower = s.lower()
                    word_overlap = len(set(s_lower.split()) & query_words)
                    # Bonus for having dates or numbers (factual content)
                    has_numbers = 1 if any(c.isdigit() for c in s) else 0
                    # Penalty for being too short
                    length_score = min(len(s) / 100, 1)
                    return word_overlap + has_numbers + length_score

                best_sentence = max(
                    sentences[:4],  # Check first 4 sentences
                    key=score_sentence,
                    default=sentences[0],
                )
                # Clean up the sentence
                best_sentence = best_sentence.strip()
                # Ensure it ends with proper punctuation
                if best_sentence and best_sentence[-1] not in ".!?":
                    best_sentence += "."
                response += best_sentence
            else:
                # Truncate at sentence boundary
                truncated = first_result[:200]
                last_punct = max(
                    truncated.rfind(". "), truncated.rfind("? "), truncated.rfind("! ")
                )
                if last_punct > 50:
                    response += truncated[: last_punct + 1]
                else:
                    response += truncated + "."
        else:
            response += (
                "Based on the available data, no definitive conclusion can be drawn."
            )

        return response

    def run_query_generator(self, query: str) -> Generator[str, None, None]:
        """Main query processor with streaming output."""
        start = time.time()

        # 1. Intent Routing & Context Retrieval
        context_data: List[str] = []
        source_label = "LIVE WEB"
        sources_metadata: List[Dict] = []

        # Simple keyword detection for Gmail intent
        if any(kw in query.lower() for kw in ["gmail", "email", "inbox"]):
            source_label = "GMAIL"
            clean_q = self._extract_keywords(query)
            raw_results = self.gmail.get_relevant_emails(clean_q)
            context_data = raw_results

            for i, txt in enumerate(raw_results):
                sources_metadata.append(
                    {
                        "title": f"Email Result {i + 1}",
                        "url": "https://mail.google.com",
                        "snippet": txt[:100] + "...",
                    }
                )
        else:
            # Default to Web Search
            clean_q = self._clean_web_query(query)
            raw_results = self.search_web(clean_q)
            context_data = [r.get("body", "") for r in raw_results if "body" in r]

            for r in raw_results:
                sources_metadata.append(
                    {
                        "title": r.get("title", "Web Result"),
                        "url": r.get("href", "#"),
                        "snippet": r.get("body", "")[:100] + "...",
                    }
                )

        # 2. Update Memory & Train VAE
        if context_data:
            self.update_memory(context_data, source=source_label)

        if self.memory_embeddings is None:
            yield (
                json.dumps(
                    {
                        "type": "token",
                        "content": f"[{source_label}] No data found. Please check connections.",
                    }
                )
                + "\n"
            )
            yield (
                json.dumps(
                    {
                        "type": "meta",
                        "stats": {"latency": "0ms"},
                        "sources": [],
                        "actions": [],
                    }
                )
                + "\n"
            )
            return

        # 3. Embed Query
        query_vec = self.embedder.encode(query)

        # 4. Generate Reasoning via VAE
        reasoned_text = self.get_reasoning_response(
            query_vec, query, forced_context=context_data
        )

        # 5. Output Formatting based on Mode
        if self.mode == "wearable":
            # Wearable Mode: Return pure JSON data
            clean_summary = (
                reasoned_text.replace("**Analysis**:", "")
                .replace("**Evidence**:", "")
                .replace("**Conclusion**:", "")
                .strip()
            )
            wearable_payload = {
                "intent": source_label,
                "summary": clean_summary,
                "timestamp": time.time(),
            }
            json_str = json.dumps(wearable_payload, indent=2)
            full_response = "```json\n" + json_str + "\n```"

            # Stream character by character for typing effect
            chunk_size = 3
            for i in range(0, len(full_response), chunk_size):
                chunk = full_response[i : i + chunk_size]
                yield json.dumps({"type": "token", "content": chunk}) + "\n"
                time.sleep(0.03)
        else:
            # Assistant Mode: Conversational
            final_output = (
                f"Hey, checking {source_label} for you...\n\n"
                f"{reasoned_text}\n\nHope that helps!"
            )

            # Stream character by character for typing effect
            chunk_size = 3
            for i in range(0, len(final_output), chunk_size):
                chunk = final_output[i : i + chunk_size]
                yield json.dumps({"type": "token", "content": chunk}) + "\n"
                time.sleep(0.03)

        # Final stats & Metadata
        latency = (time.time() - start) * 1000
        logger.info("query_processed", latency_ms=f"{latency:.2f}", source=source_label)

        yield (
            json.dumps(
                {
                    "type": "meta",
                    "stats": {"latency": f"{latency:.2f}ms"},
                    "sources": sources_metadata,
                    "actions": ["copy", "thumbs_up", "thumbs_down"],
                }
            )
            + "\n"
        )
