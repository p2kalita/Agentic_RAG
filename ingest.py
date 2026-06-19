"""
ingest.py

Loads the support-ticket CSV, turns each row into a searchable document,
embeds it with a local sentence-transformer model, and builds/saves a
FAISS index to disk so the crew doesn't have to re-embed on every run.

Expected CSV columns (case-insensitive, extra columns are ignored):
    Body, Department, Priority, Tags

Run this once whenever your CSV changes:
    python ingest.py
"""

import os
import json
import pickle
from pathlib import Path

import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = os.getenv("CSV_PATH", "tickets.csv")
INDEX_DIR = Path("faiss_index")
INDEX_DIR.mkdir(exist_ok=True)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, runs on CPU


def load_tickets(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize column names so "body"/"Body"/"BODY" all work
    df.columns = [c.strip() for c in df.columns]
    required = {"Body", "Department", "Priority", "Tags"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    df = df.dropna(subset=["Body"]).reset_index(drop=True)
    return df


def build_corpus_text(row: pd.Series) -> str:
    """
    What actually gets embedded. We fold metadata into the text itself
    (lightweight 'metadata-aware' retrieval) so semantic search can pick
    up on department/priority/tags too, not just the raw complaint text.
    """
    tags = row.get("Tags", "")
    if isinstance(tags, str):
        try:
            tags_list = json.loads(tags.replace("'", '"'))
            tags = ", ".join(tags_list)
        except Exception:
            tags = tags.strip("[]").replace("'", "")
    return (
        f"Department: {row.get('Department', 'Unknown')} | "
        f"Priority: {row.get('Priority', 'Unknown')} | "
        f"Tags: {tags}\n"
        f"Ticket: {row['Body']}"
    )


def main():
    print(f"Loading tickets from {CSV_PATH} ...")
    df = load_tickets(CSV_PATH)
    print(f"Loaded {len(df)} tickets.")

    texts = [build_corpus_text(row) for _, row in df.iterrows()]

    print(f"Embedding with {EMBED_MODEL_NAME} (CPU, may take a bit for large CSVs)...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = embedder.encode(
        texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine sim
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "tickets.index"))

    # store the original rows + text alongside the index so we can
    # return full context (not just raw vectors) at query time
    with open(INDEX_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(
            {
                "texts": texts,
                "records": df.to_dict(orient="records"),
                "embed_model": EMBED_MODEL_NAME,
            },
            f,
        )

    print(f"Done. Index saved to {INDEX_DIR}/tickets.index ({len(texts)} vectors).")


if __name__ == "__main__":
    main()