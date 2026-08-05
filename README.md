# Hotel Search Engine

A semantic search project that finds hotel reviews most similar to a natural-language query.

Given a query like *"An affordable hotel with a view of the Eiffel Tower"*, the project embeds the text and compares it against hotel review embeddings using cosine similarity, then returns the top-k most relevant matches.

## How it works

1. Load hotel review data from Hugging Face
2. Filter and clean reviews (currently Paris only)
3. Encode the query and reviews with [Sentence Transformers](https://www.sbert.net/)
4. Compute cosine similarity between the query and all review embeddings
5. Return the top-k most similar reviews with their similarity scores

## Tech stack

- [Sentence Transformers](https://www.sbert.net/) — text embeddings (`all-MiniLM-L6-v2`)
- [NumPy](https://numpy.org/) — cosine similarity from scratch
- [FAISS](https://github.com/facebookresearch/faiss) — fast approximate nearest-neighbor search
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/) — hotel review dataset
- [pandas](https://pandas.pydata.org/) — filtering and data cleaning

## Getting started

### Prerequisites

- Python 3.9+

### Installation

```bash
git clone <your-repo-url>
cd hotel-search-engine

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

For GPU-accelerated FAISS, replace `faiss-cpu` with `faiss-gpu` in `requirements.txt` (requires a CUDA-compatible setup).

### Run the search

NumPy (from scratch):

```bash
python top_k_cosine_similarity_from_scratch.py
```

FAISS:

```bash
python top_k_cosine_similarity_faiss.py
```

On the first run, each script downloads the embedding model and dataset from Hugging Face. They then search Paris hotel reviews and print the top 5 matches.

### Example output

Query:

```text
An affordable hotel with view of the Eiffel Tower
```

![Terminal output showing top hotel search results](https://i.ibb.co/BHSfshnk/Screenshot-2026-07-30-at-03-18-22.png)

## Dataset

Reviews come from the [traversaal-ai-hackathon/hotel_datasets](https://huggingface.co/datasets/traversaal-ai-hackathon/hotel_datasets) dataset on Hugging Face.

## Todos

- [x] FAISS-based search for faster retrieval at scale
- [ ] Support different cities and custom search queries
- [ ] Cache embeddings to avoid recomputing them on every run
- [ ] Reuse cached embeddings between searches
- [ ] Add a simple user interface
- [ ] Add an LLM step that summarizes retrieved reviews into a structured recommendation
- [ ] Choose a license for the project
- [ ] Deploy the project

## Status

🚧 Work in progress — both NumPy and FAISS search implementations are working; UI and caching features are planned.
