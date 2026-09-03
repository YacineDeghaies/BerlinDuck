"""Query-time retrieval: embed a query and search a vector store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from berlinduck.embeddings import Embedder
from berlinduck.vectorstore import Document, FaissStore, VectorStore


@dataclass(frozen=True)
class SearchHit:
    id: str
    score: float
    text: str
    metadata: dict[str, Any]


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder | None = None) -> None:
        self.store = store
        self.embedder = embedder or Embedder()

    @classmethod
    def from_index(cls, path: str | Path, embedder: Embedder | None = None) -> "Retriever":
        """Load a persisted FAISS index built by :mod:`berlinduck.ingest`."""
        return cls(FaissStore.load(path), embedder)

    @classmethod
    def from_documents(
        cls, documents: list[Document], embedder: Embedder | None = None
    ) -> "Retriever":
        """Build an ephemeral in-memory index (used by the CLI demo and tests)."""
        embedder = embedder or Embedder()
        store = FaissStore(dimension=embedder.dimension)
        store.add(embedder.encode_documents([doc.text for doc in documents]), documents)
        return cls(store, embedder)

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        query_embedding = self.embedder.encode_query(query)
        return [
            SearchHit(
                id=result.document.id,
                score=result.score,
                text=result.document.text,
                metadata=result.document.metadata,
            )
            for result in self.store.search(query_embedding, k)
        ]
