# BerlinDuck

A semantic search engine for hotel reviews, implemented in Python 3.

Given a query like *"An affordable hotel with a view of the Eiffel Tower"*, the code embeds the text and compares it against hotel review embeddings using similarity search, then returns the top-k most relevant matches.

## Project layout

```
src/berlinduck/
  similarity.py   L2 normalization + a from-scratch cosine-search reference impl (NumPy)
  chunking.py     split review text into overlapping windows
  embeddings.py   Embedder: SentenceTransformer wrapper, normalized vectors
  vectorstore.py  QdrantStore: embedded (in-memory / on-disk) or a Qdrant server
  data.py         load + clean the hotel-review dataset
  ingest.py       pipeline: load -> chunk -> embed -> upsert to Qdrant
  retriever.py    Retriever: embed a query, search Qdrant, return SearchHits
  demo.py         CLI entry point
conftest.py       puts src/ on sys.path for pytest (no install step)
requirements.txt  dependencies (pip)
```

## Setup

Requires **Python 3.11+**. Dependencies are listed in `requirements.txt` and
installed with pip. If the system Python is older (common on shared servers),
get 3.11 via conda without needing root:

```bash
conda create -n berlinduck python=3.11
conda activate berlinduck
pip install -r requirements.txt
```

Or, on a machine that already has a 3.11+ interpreter:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The package is not installed — run it with `src/` on the import path
(`PYTHONPATH=src`). `pytest` handles this itself via `conftest.py`.

## Usage

```bash
# 1. ingest: embed reviews and upsert them into an embedded Qdrant at ./data/qdrant
PYTHONPATH=src python -m berlinduck.ingest --locality Paris

# 2. query it
PYTHONPATH=src python -m berlinduck.demo "a quiet hotel near the Louvre" -k 5

# tests (similarity + chunking + vector store; need only numpy + qdrant-client)
pytest
```

To use a running Qdrant server instead of the embedded one, pass
`--qdrant-url http://localhost:6333` to both commands. If no store exists at
`data/qdrant`, `demo.py` builds an in-memory one on the fly.

## Tech stack

- [Sentence Transformers](https://www.sbert.net/) — text embeddings (`all-MiniLM-L6-v2`)
- [Qdrant](https://qdrant.tech/) — vector database (embedded or server), cosine distance
- [NumPy](https://numpy.org/) — cosine similarity reference implementation
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/) — hotel review dataset
- [pandas](https://pandas.pydata.org/) — filtering and data cleaning

## Embedding Model

The project uses:

```python

SentenceTransformer("all-MiniLM-L6-v2")

```

This model converts text into dense vector embeddings suitable for semantic search.

## Dataset

Reviews come from the [traversaal-ai-hackathon/hotel_datasets](https://huggingface.co/datasets/traversaal-ai-hackathon/hotel_datasets) dataset on Hugging Face.

This repo will still grow

