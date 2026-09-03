"""Ingestion pipeline: load reviews -> clean -> chunk -> embed -> persist an index.

Run once, offline, before serving queries:

    python -m berlinduck.ingest --locality Paris --out data/index
"""

from __future__ import annotations

import argparse
from pathlib import Path

from berlinduck.chunking import chunk_text
from berlinduck.data import load_reviews
from berlinduck.embeddings import Embedder
from berlinduck.vectorstore import Document, FaissStore, NumpyStore

DEFAULT_INDEX_DIR = Path("data/index")
_BACKENDS = {"faiss": FaissStore, "numpy": NumpyStore}


def build_index(
    locality: str | None = "Paris",
    out_dir: str | Path = DEFAULT_INDEX_DIR,
    chunk_size: int = 512,
    overlap: int = 64,
    backend: str = "faiss",
    embedder: Embedder | None = None,
) -> FaissStore | NumpyStore:
    df = load_reviews(locality=locality)
    embedder = embedder or Embedder()

    documents: list[Document] = []
    for review_id, row in df.iterrows():
        for chunk_index, chunk in enumerate(chunk_text(row["review_text"], chunk_size, overlap)):
            documents.append(
                Document(
                    id=f"{review_id}-{chunk_index}",
                    text=chunk,
                    metadata={
                        "review_id": int(review_id),
                        "chunk_index": chunk_index,
                        "hotel_name": row.get("hotel_name"),
                        "locality": row.get("locality"),
                        "hotel_description": row.get("hotel_description"),
                    },
                )
            )
    if not documents:
        raise ValueError(f"no reviews found for locality={locality!r}")

    embeddings = embedder.encode_documents([doc.text for doc in documents])
    store = _BACKENDS[backend](dimension=embedder.dimension)
    store.add(embeddings, documents)
    store.persist(out_dir)
    print(f"indexed {len(store)} chunks from {len(df)} reviews -> {out_dir} ({backend})")
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locality", default="Paris")
    parser.add_argument("--out", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--backend", choices=sorted(_BACKENDS), default="faiss")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()
    build_index(
        locality=args.locality,
        out_dir=args.out,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
