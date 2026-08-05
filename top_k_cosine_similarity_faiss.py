import time

import faiss
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def main() -> None:
    dataset = load_dataset("traversaal-ai-hackathon/hotel_datasets")

    df = pd.DataFrame(dataset["train"])
    df_paris = df.loc[
        (df["locality"] == "Paris")
        & df["review_text"].notna()
    ].reset_index(drop=True)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    if torch.cuda.is_available():
        model = model.to("cuda")
        print("CUDA is available. The model has been moved to the GPU.")
    else:
        print("CUDA is not available. The model will run on the CPU.")

    reviews = df_paris["review_text"].tolist()
    query = "Hotel near the Louvre with great food nearby."

    review_embeddings = model.encode(
        reviews,
        show_progress_bar=True,
    ).astype(np.float32)

    query_embedding = model.encode(
        [query]
    ).astype(np.float32)

    # Normalize vectors so inner product equals cosine similarity.
    faiss.normalize_L2(review_embeddings)
    faiss.normalize_L2(query_embedding)

    embedding_dimension = review_embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(review_embeddings)

    k = 5

    start_time = time.time()
    similarity_scores, indices = index.search(query_embedding, k)
    faiss_search_time = time.time() - start_time

    print(f"FAISS search time: {faiss_search_time:.4f} seconds")
    print(f"Query: {query}")
    print("Top hotels with similar reviews using FAISS:")

    for rank, (idx, score) in enumerate(
        zip(indices[0], similarity_scores[0]),
        start=1,
    ):
        row = df_paris.iloc[idx]

        print(f"{rank}. {row['hotel_name']}")
        print(f"Review: {row['review_text']}")
        print(f"Cosine similarity: {score:.4f}")
        print()


if __name__ == "__main__":
    main()
