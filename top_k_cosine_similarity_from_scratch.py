# Install & import all required libraries
from typing import Any
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pandas as pd


def cosine_similarity(query_embedding,
                      review_embeddings,
                      k):
    # Goal: returns the top-k most relevant documents along with their indices
    
    # Ensure we have consistent format and numpy array data types
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    review_embeddings = np.asarray(review_embeddings, dtype=np.float32)
    
    # Do some check over our passed arguments
        # check dimensions 
    if query_embedding.ndim != 1:
        raise ValueError(
            "query embedding needs to be one-dimensional"
        )
        
    if review_embeddings.ndim != 2:
        raise ValueError(
            "review embeddings need to be two-dimensional"
        )
        
    if review_embeddings.shape[1] != query_embedding.size:
        raise ValueError(
            "Embedding dimensions for not match"
            f"The query embedding has dimension {query_embedding.shape} while "
            f"the reviews have dimensions {review_embeddings.shape}"
        )
        
    # Check for valid k value
    if (
        isinstance(k, (bool, np.bool_) ) or # to guard against k=True or k=False
        not isinstance(k, (int, np.integer)) or  # to guard against k="5" or k=5.4
        k < 1 or 
        k > len(review_embeddings)
    ):
        raise ValueError(
            f"k must be between 1 and {len(review_embeddings)}"
        )
        
    # Compute dot products: (m, n) * (n,)
    dot_products = review_embeddings @ query_embedding
    
    # Compute Norms 
    query_embedding_norm = np.linalg.norm(query_embedding)

        # review_embeddings is a 2-D Matrix, we need to specify across which Axis we want to normalize
    review_embeddings_norm = np.linalg.norm(review_embeddings, axis=1)

    denominator = query_embedding_norm * review_embeddings_norm
    
    # Compute Cosine Similarity 
    # one needs to have the ability to know the kind of unsual values that may occur and may result in an error
    cosine_scores = np.divide(
        dot_products,
        denominator,
        # maybe one of the reviews was zero-vector or had a zero value 
        where=(denominator != 0),
        out=np.zeros_like(dot_products, dtype=np.float32)
    )

    indices = np.argsort(cosine_scores)[::-1][:k]
    scores = cosine_scores[indices]
    
    # Return the indices and scores of the top-k most relevant documents
    return indices, scores
    

def main():
    
    # Define search query
    user_query = "An affordable hotel with view of the Eiffel Tower"

    # Define the top-k most relevant results
    k = 5

    # Load dataset
    dataset = load_dataset("traversaal-ai-hackathon/hotel_datasets")

    # Extract the 'train' split and convert it into a DataFrame
    df = dataset["train"].to_pandas()
    
    # Filter for hotels in Paris only
    df_paris = df.loc[df.locality == "Paris"]
    
    # Clean the DataFrame
    df_paris = df_paris.dropna(subset=["review_text"])

    # Extract reviews as a list of strings
    reviews = df_paris.review_text.tolist()
    
    # Load an embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Encode the user query
    query_embedding = model.encode(user_query)
    
    # Encode reviews
    review_embeddings = model.encode(reviews, show_progress_bar=True)

    # Run Cosine Search
    indices, scores = cosine_similarity(query_embedding, review_embeddings, k)
    
    # Print top-k most relevant candiate documents
    for i, (idx, score) in enumerate[tuple[Any, Any]](zip(indices, scores), 1):
        print()
        print()
        print(f"Query: {user_query}.")
        print("Top hotels with similar reviews:")
        print(f"{i}. {df_paris.iloc[idx].hotel_name}")
        print(f"Review: {df_paris.iloc[idx].review_text}")
        print(f"Similarity: {score}")
        print()
        
        

if __name__ == "__main__":
    main()