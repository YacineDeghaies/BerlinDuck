# BerlinDuck

A semantic search engine for hotel reviews, implemented in Python 3.

Given a query like *"An affordable hotel with a view of the Eiffel Tower"*, the code embeds the text and compares it against hotel review embeddings using similarity search, then returns the top-k most relevant matches.

## Project layout

```
src/berlinduck/
  similarity.py   from-scratch cosine top-k search + L2 normalization (NumPy only)
  chunking.py     split review text into overlapping windows
  embeddings.py   Embedder: SentenceTransformer wrapper, normalized vectors
  vectorstore.py  VectorStore protocol + NumpyStore / FaissStore backends (persistent)
  data.py         load + clean the hotel-review dataset
  ingest.py       pipeline: load -> chunk -> embed -> persist an index
  retriever.py    Retriever: embed a query, search a store, return SearchHits
  demo.py         CLI entry point
tests/            unit tests (similarity, chunking, vector stores)
top_k_cosine_similarity_faiss.ipynb   the same pipeline in a notebook
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# 1. build a persistent index (writes data/index/: vectors + documents.jsonl + meta.json)
python -m berlinduck.ingest --locality Paris --backend faiss

# 2. query it
python -m berlinduck.demo "a quiet hotel near the Louvre" -k 5

# tests (the similarity + chunking + store tests need only numpy + faiss-cpu)
pytest
```

`--backend numpy` swaps the FAISS index for the from-scratch cosine search in
`similarity.py`; both implement the same `VectorStore` interface. If no index
exists, `demo.py` builds an in-memory one on the fly.

## Tech stack

- [Sentence Transformers](https://www.sbert.net/) — text embeddings (`all-MiniLM-L6-v2`)
- [NumPy](https://numpy.org/) — cosine similarity from scratch
- [FAISS](https://github.com/facebookresearch/faiss) — fast approximate nearest-neighbor search
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/) — hotel review dataset
- [pandas](https://pandas.pydata.org/) — filtering and data cleaning

## Embedding Model

The project uses:

```python

SentenceTransformer("all-MiniLM-L6-v2")

```

This model converts text into dense vector embeddings suitable for semantic search.
For GPU-accelerated FAISS, replace `faiss-cpu` with `faiss-gpu` in `pyproject.toml` (requires a CUDA-compatible setup).

## Dataset

Reviews come from the [traversaal-ai-hackathon/hotel_datasets](https://huggingface.co/datasets/traversaal-ai-hackathon/hotel_datasets) dataset on Hugging Face.

This repo will still grow

