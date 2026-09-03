"""Vector-store abstraction with two interchangeable backends.

``NumpyStore`` uses the from-scratch cosine search in :mod:`berlinduck.similarity`;
``FaissStore`` uses a FAISS flat inner-product index. Both persist to a directory
holding the vectors, a ``documents.jsonl`` sidecar, and a ``meta.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from berlinduck.similarity import top_k_cosine


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredDocument:
    document: Document
    score: float


class VectorStore(Protocol):
    dimension: int

    def add(self, embeddings: np.ndarray, documents: list[Document]) -> None: ...
    def search(self, embedding: np.ndarray, k: int) -> list[ScoredDocument]: ...
    def persist(self, path: str | Path) -> None: ...
    def __len__(self) -> int: ...


# --- shared sidecar helpers -------------------------------------------------


def _write_sidecar(path: Path, dimension: int, documents: list[Document]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "documents.jsonl").open("w") as fh:
        for doc in documents:
            fh.write(json.dumps({"id": doc.id, "text": doc.text, "metadata": doc.metadata}) + "\n")
    (path / "meta.json").write_text(json.dumps({"dimension": dimension}))


def _read_sidecar(path: Path) -> tuple[int, list[Document]]:
    dimension = json.loads((path / "meta.json").read_text())["dimension"]
    documents: list[Document] = []
    with (path / "documents.jsonl").open() as fh:
        for line in fh:
            record = json.loads(line)
            documents.append(Document(record["id"], record["text"], record["metadata"]))
    return dimension, documents


def _validate(embeddings: np.ndarray, documents: list[Document], dimension: int) -> np.ndarray:
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != dimension:
        raise ValueError(f"expected embeddings of shape (n, {dimension}), got {embeddings.shape}")
    if len(documents) != embeddings.shape[0]:
        raise ValueError("number of documents must match number of embeddings")
    return embeddings


# --- backends -------------------------------------------------------------


class NumpyStore:
    """Persistent store backed by the from-scratch cosine search."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.documents: list[Document] = []
        self._embeddings: np.ndarray | None = None

    def add(self, embeddings: np.ndarray, documents: list[Document]) -> None:
        embeddings = _validate(embeddings, documents, self.dimension)
        self._embeddings = (
            embeddings
            if self._embeddings is None
            else np.vstack([self._embeddings, embeddings])
        )
        self.documents.extend(documents)

    def search(self, embedding: np.ndarray, k: int) -> list[ScoredDocument]:
        if self._embeddings is None:
            raise RuntimeError("store is empty")
        k = min(k, len(self.documents))
        indices, scores = top_k_cosine(np.asarray(embedding, dtype=np.float32), self._embeddings, k)
        return [
            ScoredDocument(self.documents[int(i)], float(s)) for i, s in zip(indices, scores)
        ]

    def __len__(self) -> int:
        return len(self.documents)

    def persist(self, path: str | Path) -> None:
        path = Path(path)
        _write_sidecar(path, self.dimension, self.documents)
        np.save(path / "embeddings.npy", self._embeddings)

    @classmethod
    def load(cls, path: str | Path) -> "NumpyStore":
        path = Path(path)
        dimension, documents = _read_sidecar(path)
        store = cls(dimension)
        store.documents = documents
        store._embeddings = np.load(path / "embeddings.npy")
        return store


class FaissStore:
    """Persistent FAISS flat inner-product index (cosine, given normalized inputs)."""

    def __init__(self, dimension: int) -> None:
        import faiss

        self._faiss = faiss
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: list[Document] = []

    def add(self, embeddings: np.ndarray, documents: list[Document]) -> None:
        embeddings = _validate(embeddings, documents, self.dimension)
        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, embedding: np.ndarray, k: int) -> list[ScoredDocument]:
        if not self.documents:
            raise RuntimeError("store is empty")
        k = min(k, len(self.documents))
        query = np.ascontiguousarray(embedding, dtype=np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, k)
        return [
            ScoredDocument(self.documents[i], float(s))
            for s, i in zip(scores[0], indices[0])
            if i != -1
        ]

    def __len__(self) -> int:
        return len(self.documents)

    def persist(self, path: str | Path) -> None:
        path = Path(path)
        _write_sidecar(path, self.dimension, self.documents)
        self._faiss.write_index(self.index, str(path / "index.faiss"))

    @classmethod
    def load(cls, path: str | Path) -> "FaissStore":
        path = Path(path)
        dimension, documents = _read_sidecar(path)
        store = cls(dimension)
        store.index = store._faiss.read_index(str(path / "index.faiss"))
        store.documents = documents
        return store
