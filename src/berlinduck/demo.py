"""CLI demo: semantic search over hotel reviews.

Uses a persisted index at ``data/index`` if present, otherwise builds an
in-memory one on the fly:

    python -m berlinduck.ingest              # optional: build data/index first
    python -m berlinduck.demo "a quiet hotel near the Louvre" -k 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from berlinduck.data import load_reviews
from berlinduck.retriever import Retriever
from berlinduck.vectorstore import Document

DEFAULT_QUERY = "An affordable hotel with a view of the Eiffel Tower"
DEFAULT_INDEX_DIR = Path("data/index")


def _load_retriever(index_dir: Path, locality: str) -> Retriever:
    if (index_dir / "meta.json").exists():
        print(f"loading index from {index_dir}")
        return Retriever.from_index(index_dir)

    print(f"no index at {index_dir}; building an in-memory one for locality={locality!r}")
    df = load_reviews(locality=locality)
    documents = [
        Document(
            id=str(review_id),
            text=row["review_text"],
            metadata={"hotel_name": row.get("hotel_name"), "locality": row.get("locality")},
        )
        for review_id, row in df.iterrows()
    ]
    return Retriever.from_documents(documents)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument("-k", type=int, default=5, help="number of results")
    parser.add_argument("--locality", default="Paris")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    args = parser.parse_args()

    retriever = _load_retriever(args.index_dir, args.locality)
    hits = retriever.search(args.query, k=args.k)

    print(f"\nQuery: {args.query}\n")
    for rank, hit in enumerate(hits, start=1):
        hotel = hit.metadata.get("hotel_name", "?")
        print(f"{rank}. {hotel}  (similarity {hit.score:.3f})")
        print(f"   {hit.text}\n")


if __name__ == "__main__":
    main()
