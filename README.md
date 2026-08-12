# Hotel Search Engine

A bunch of notebooks demonstrating how to build a semantic search engine for hotel reviews implemented in Python 3.

Given a query like *"An affordable hotel with a view of the Eiffel Tower"*, this notebook embeds the text and compares it against hotel review embeddings using similarity search, then returns the top-k most relevant matches.

## Tech stack

- [Sentence Transformers](https://www.sbert.net/) — text embeddings (`all-MiniLM-L6-v2`)
- [NumPy](https://numpy.org/) — cosine similarity from scratch
- [FAISS](https://github.com/facebookresearch/faiss) — fast approximate nearest-neighbor search
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/) — hotel review dataset
- [pandas](https://pandas.pydata.org/) — filtering and data cleaning

## Embedding Model

The notebook uses:

```python

SentenceTransformer("all-MiniLM-L6-v2")

```

This model converts text into dense vector embeddings suitable for semantic search.
For GPU-accelerated FAISS, replace `faiss-cpu` with `faiss-gpu` in `requirements.txt` (requires a CUDA-compatible setup).

## Dataset

Reviews come from the [traversaal-ai-hackathon/hotel_datasets](https://huggingface.co/datasets/traversaal-ai-hackathon/hotel_datasets) dataset on Hugging Face.