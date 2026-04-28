from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import hashlib
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


@dataclass(frozen=True)
class BotPersona:
    # Simple immutable container so personas can be passed around safely without mutation.
    bot_id: str
    description: str


def get_default_personas() -> List[BotPersona]:
    """Return the built-in personas used throughout the demo pipeline."""
    return [
        BotPersona(
            bot_id="tech_maximalist",
            description="I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration. I dismiss regulatory concerns.",
        ),
        BotPersona(
            bot_id="doomer_skeptic",
            description="I believe late-stage capitalism and tech monopolies are destroying society. I am highly critical of AI, social media, and billionaires. I value privacy and nature.",
        ),
        BotPersona(
            bot_id="finance_bro",
            description="I strictly care about markets, interest rates, trading algorithms, and making money. I speak in finance jargon and view everything through the lens of ROI.",
        ),
    ]


def _unit_normalize(vecs: np.ndarray) -> np.ndarray:
    """Normalize vectors so cosine similarity can be computed with a dot product."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


class _LocalHashEmbeddings:
    """
    Offline fallback when sentence-transformers model download is blocked.
    Produces deterministic vectors via token hashing; good enough for a demo router.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        # Hash tokens into a fixed-size vector so the demo can still run without downloading
        # a real embedding model. This preserves deterministic behavior for local testing.
        v = np.zeros(self.dim, dtype=np.float32)
        tokens = [t for t in text.lower().split() if t]
        for tok in tokens:
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if (h[4] % 2 == 0) else -1.0
            v[idx] += sign
        # normalize to unit length
        n = float(np.linalg.norm(v)) or 1.0
        return (v / n).astype(float).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple persona descriptions for indexing."""
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single user post using the same hashing scheme as documents."""
        return self._embed(text)


class VectorPersonaRouter:
    """
    Phase 1: Embed bot personas, store in an in-memory vector store (FAISS),
    and route posts to bots based on cosine similarity.
    """

    def __init__(
        self,
        personas: Sequence[BotPersona] | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        # Default personas keep the router self-contained, but callers can inject their own
        # set for experiments or tests.
        self.personas: List[BotPersona] = list(personas or get_default_personas())

        try:
            # Prefer a real sentence-transformer for higher quality semantic routing.
            self._embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        except Exception:
            # If HF downloads are blocked (common in locked-down networks), fall back locally.
            self._embeddings = _LocalHashEmbeddings()

        # Convert personas into LangChain Documents so they can be stored in FAISS with
        # metadata preserving the bot ID that each embedding belongs to.
        self._docs: List[Document] = [
            Document(page_content=p.description, metadata={"bot_id": p.bot_id})
            for p in self.personas
        ]

        # In-memory FAISS store (fulfills "ChromaDB or FAISS in-memory").
        self._vs = FAISS.from_documents(self._docs, self._embeddings)

        # Also keep normalized raw persona vectors for explicit cosine routing.
        persona_vecs = np.array(self._embeddings.embed_documents([d.page_content for d in self._docs]))
        self._persona_vecs = _unit_normalize(persona_vecs)
        self._bot_ids = [d.metadata["bot_id"] for d in self._docs]

    def route_post_to_bots(self, post_content: str, threshold: float = 0.3) -> List[str]:
        """
        Returns list of matching bot IDs based on cosine similarity above threshold.

        The default threshold is 0.3 because `all-MiniLM-L6-v2` tends to produce lower
        raw cosine scores than larger embedding models, so a higher cutoff like 0.85
        would incorrectly reject semantically relevant personas in this demo.
        """
        # Embed and normalize the incoming post so its dot product with the normalized
        # persona vectors is the cosine similarity score.
        q = np.array(self._embeddings.embed_query(post_content), dtype=float).reshape(1, -1)
        qn = _unit_normalize(q)
        sims = (self._persona_vecs @ qn.T).reshape(-1)

        # Keep both the ID and the score during filtering so results can be sorted before
        # returning only the bot IDs to the caller.
        matches: List[Tuple[str, float]] = [
            (bot_id, float(sim)) for bot_id, sim in zip(self._bot_ids, sims) if sim >= threshold
        ]
        matches.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in matches]

    def debug_similarities(self, post_content: str) -> Dict[str, float]:
        """Return raw similarity scores for inspection without applying a threshold."""
        q = np.array(self._embeddings.embed_query(post_content), dtype=float).reshape(1, -1)
        qn = _unit_normalize(q)
        sims = (self._persona_vecs @ qn.T).reshape(-1)
        return {bot_id: float(sim) for bot_id, sim in zip(self._bot_ids, sims)}
