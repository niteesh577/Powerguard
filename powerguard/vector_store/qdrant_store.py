"""
vector_store/qdrant_store.py
============================
Qdrant Cloud integration for scientific literature RAG.

Phase 3: The risk_agent can call retrieve_context() to ground recommendations
in evidence from peer-reviewed powerlifting injury prevention research.

Collection: "powerguard_literature"
Embedding model: all-MiniLM-L6-v2 (384 dims, fast, free)

Usage:
    context = await retrieve_context("knee valgus squat injury risk")
    # Returns: list of relevant paper snippets
"""

from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config.settings import get_settings
from observability.logger import get_logger

logger = get_logger("qdrant_store")


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Singleton Qdrant client."""
    settings = get_settings()
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=10,
    )
    logger.info("qdrant.connected", url=settings.qdrant_url)
    return client


def ensure_collection_exists() -> None:
    """Create the literature collection if it doesn't exist."""
    settings = get_settings()
    client   = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]

    if settings.qdrant_collection not in collections:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.qdrant_embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info("qdrant.collection_created", name=settings.qdrant_collection)
    else:
        logger.info("qdrant.collection_exists", name=settings.qdrant_collection)


def _embed(text: str) -> list[float]:
    """
    Local embedding using sentence-transformers.
    Install: uv add sentence-transformers
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(text).tolist()
    except ImportError:
        logger.warning("qdrant.sentence_transformers_not_installed")
        return [0.0] * 384


async def retrieve_context(query: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve relevant scientific literature snippets for a query.
    Returns list of {text, source, score} dicts.
    Falls back gracefully if Qdrant is unavailable.
    """
    settings = get_settings()

    if not settings.qdrant_url or not settings.qdrant_api_key:
        logger.warning("qdrant.not_configured")
        return []

    try:
        client  = get_qdrant_client()
        vector  = _embed(query)
        results = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )
        snippets = [
            {
                "text":   r.payload.get("text", ""),
                "source": r.payload.get("source", ""),
                "score":  round(r.score, 3),
            }
            for r in results
            if r.score > 0.6   # relevance threshold
        ]
        logger.info("qdrant.retrieved", query=query[:50], hits=len(snippets))
        return snippets

    except Exception as e:
        logger.warning("qdrant.retrieval_failed", error=str(e))
        return []   # Non-blocking: agents work without Qdrant


def upsert_literature(documents: list[dict]) -> int:
    """
    Upsert scientific literature into the vector store.

    Each document must have:
      - id:     unique int
      - text:   full text of the passage
      - source: citation string (author, year, journal)

    Returns: number of documents upserted.
    """
    settings = get_settings()
    client   = get_qdrant_client()
    ensure_collection_exists()

    points = [
        PointStruct(
            id=doc["id"],
            vector=_embed(doc["text"]),
            payload={"text": doc["text"], "source": doc["source"]},
        )
        for doc in documents
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points)
    logger.info("qdrant.upserted", count=len(points))
    return len(points)
