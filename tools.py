"""
tools.py

Custom CrewAI tool that lets an agent semantically search the FAISS index
of support tickets built by ingest.py.
"""

import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from crewai.tools import tool

INDEX_DIR = Path("faiss_index")
_embedder = None
_index = None
_meta = None


def _load_resources():
    """Lazy-load the index/model once, on first tool call (not at import time)."""
    global _embedder, _index, _meta
    if _index is None:
        index_path = INDEX_DIR / "tickets.index"
        meta_path = INDEX_DIR / "metadata.pkl"
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                "FAISS index not found. Run `python ingest.py` first to build it "
                "from your CSV."
            )
        _index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            _meta = pickle.load(f)
        _embedder = SentenceTransformer(_meta["embed_model"])
    return _embedder, _index, _meta


@tool("Ticket Knowledge Base Search")
def search_tickets(query: str, top_k: int = 3) -> str:
    """
    Searches the support-ticket knowledge base for tickets semantically
    similar to the query. Use this whenever you need to find past tickets,
    examples of how similar issues were categorized, or context before
    answering a question about support data (e.g. department, priority,
    tags, common issue types).

    Args:
        query: The natural-language question or issue description to search for.
        top_k: How many similar tickets to return (default 3).

    Returns:
        A formatted string with the top matching tickets, including their
        department, priority, tags, and body text.
    """
    top_k = 3
    embedder, index, meta = _load_resources()

    query_vec = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    scores, indices = index.search(query_vec, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue
        record = meta["records"][idx]
        results.append(
            f"[Result {rank} | similarity={score:.3f}]\n"
            f"Department: {record.get('Department', 'Unknown')}\n"
            f"Priority: {record.get('Priority', 'Unknown')}\n"
            f"Tags: {record.get('Tags', '')}\n"
            f"Body: {record.get('Body', '')}\n"
        )

    if not results:
        return "No matching tickets found in the knowledge base."

    return "\n---\n".join(results)