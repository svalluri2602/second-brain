"""
Vector Store — Qdrant layer for semantic search.

Handles:
  - Ingesting CV (cv.md) into Qdrant as chunks
  - Ingesting any text (notes, articles, scraped content)
  - Semantic search over stored chunks
  - Outreach message generation from relevant chunks

Usage:
  python3 -m agent.knowledge.vector_store          # ingest cv.md
  python3 -m agent.knowledge.vector_store search "LangGraph experience"
"""

import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding

load_dotenv()

client   = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
qdrant   = QdrantClient(host="localhost", port=6333)
embedder = TextEmbedding()

ROOT     = Path(__file__).parent.parent.parent
CV_PATH  = ROOT / "career-ops" / "cv.md"

COLLECTION     = "resume_chunks"
CHUNK_SIZE     = 500   # words per chunk
VECTOR_SIZE    = 384


# ── Qdrant setup ──────────────────────────────────────────────────────────────

def ensure_collection():
    if not qdrant.collection_exists(COLLECTION):
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )


# ── Ingestion ─────────────────────────────────────────────────────────────────

def _chunk_text(text: str, source: str) -> list[dict]:
    words  = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        chunks.append({"text": chunk, "source": source})
    return chunks


def _embed_and_store(chunks: list[dict]):
    ensure_collection()
    texts  = [c["text"] for c in chunks]
    embeddings = list(embedder.embed(texts))
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i].tolist(),
            payload={"text": chunks[i]["text"], "source": chunks[i]["source"]}
        )
        for i in range(len(chunks))
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)
    return len(points)


def ingest_cv():
    if not CV_PATH.exists():
        print(f"cv.md not found at {CV_PATH}")
        return 0
    text   = CV_PATH.read_text(encoding="utf-8")
    chunks = _chunk_text(text, source="cv.md")
    n      = _embed_and_store(chunks)
    print(f"CV ingested: {n} chunks → Qdrant")
    return n


def ingest_text(text: str, source: str = "note") -> int:
    chunks = _chunk_text(text, source=source)
    n      = _embed_and_store(chunks)
    print(f"Ingested '{source}': {n} chunks → Qdrant")
    return n


def ingest_file(path: str | Path) -> int:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    return ingest_text(text, source=path.name)


# ── Search ────────────────────────────────────────────────────────────────────

def search_relevant(query: str, limit: int = 3) -> list[str]:
    ensure_collection()
    embedding = list(embedder.embed([query]))[0]
    results   = qdrant.query_points(
        collection_name=COLLECTION,
        query=embedding.tolist(),
        limit=limit
    ).points
    return [r.payload["text"] for r in results]


# ── Outreach generation ───────────────────────────────────────────────────────

def generate_outreach(
    chunks: list[str],
    job_description: str,
    target_name: str,
    sender_name: str
) -> str:
    context  = "\n".join(chunks)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""Write a personalized LinkedIn outreach message.

Sender background (from CV):
{context}

Target role:
{job_description}

Sender: {sender_name}
Target: {target_name}

One message, under 300 characters. Natural, not salesy. Reference one specific experience."""
        }]
    )
    return response.content[0].text.strip()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        query   = " ".join(sys.argv[2:]) or "AI agent experience"
        results = search_relevant(query)
        print(f"\nTop {len(results)} chunks for: '{query}'\n")
        for i, r in enumerate(results, 1):
            print(f"── {i} ──────────────────────────")
            print(r[:400])
            print()
    else:
        ingest_cv()
