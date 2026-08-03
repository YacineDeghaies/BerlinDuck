import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pandas as pd

def cosine_similarity(query_embedding, review_embeddings, k):
    
    # Make sure the we have consistent format and Numpy array data types
    query_embedding = np.asarray(query_embedding, dtype=float)
    review_embeddings = np.asarray(review_embeddings, dtype=float)
    
    # First, check if they both have the correct dimensions before computing dot products
    if query_embedding.ndim != 1:
        raise ValueError("query_embedding should be one-dimensional")
   
    if review_embeddings.ndim != 2:
        raise ValueError("the argument review_embeddings should be two-dimensional")
    
    # Check if dimensions (query_embedding,) x (review_embeddings, query_embedding) already match
    if query_embedding.size != review_embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimensions do not match. "
            f"The query has dimensions {query_embedding.size}"
            f"while reviews have dimensions {review_embeddings.shape[1]}"
        )
        
    # Check if top-k recall value is valid
    if (
        isinstance(k, (bool, np.bool_)) or 
        not isinstance(k, (int, np.integer)) or 
        k < 1 or 
        k > len(review_embeddings)
    ):
        raise ValueError(
            f"K must be between 1 and {len(review_embeddings)}"
        )
        
    # Compute dot products
    dot_products = review_embeddings @ query_embedding
    
    # Compute one Norm for the query embedding
    query_embedding_norm = np.linalg.norm(query_embedding)

    # Compute one Norm for each review
    review_embeddings_norm = np.linalg.norm(review_embeddings, axis=1)
    
    denominator = query_embedding_norm * review_embeddings_norm

    # Compute Cosine Similarity
    cosine_similarity = np.divide(
        dot_products,
        denominator,
        out=np.zeros_like(dot_products, dtype=float),
        where=denominator != 0,
    )
    
    # Retrieve top-k relevant results
    top_k_indices = np.argsort(cosine_similarity)[::-1][:k]
    
    # Retrieve their cosine similarity scores
    top_k_scores = cosine_similarity[top_k_indices]
    
    return top_k_indices, top_k_scores

def main():
    
    # Define the search query
    user_query = "An affordable hotel with view of the Eiffel Tower"

    # Load the dataset 
    dataset = load_dataset("traversaal-ai-hackathon/hotel_datasets")
    
    # Extract the train split and convert it into a DataFrame
    df = dataset["train"].to_pandas()
    
    # Filter for hotels located only in Paris
    df_paris = df.loc[df.locality == "Paris"].copy()
    
    # Clean the column review_text within the DataFrame
    df_paris = df_paris.dropna(subset=["review_text"])
    
    df_paris = df_paris.loc[
        df_paris.review_text.str.strip() != "" # Checks weather each cleaned review is not an empty string
    ]
    
    # Extract reviews
    reviews = df_paris.review_text.tolist()
    
    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")
  
    # Encoder user query 
    query_embedding = model.encode(user_query)
    
    # Encoder reviews
    review_embeddings = model.encode(reviews, show_progress_bar=True)

    # Top-k similar reviews to retrieve
    k = 5 

    top_k_indices, top_k_scores = cosine_similarity(query_embedding, review_embeddings, k)
    
    # Now I could feed the retrieved result to an LLM to process, customize and format the output
    print(f"Query: {user_query}")
    print(f"The top-{k} Hotels with similar reviews:")
    for i, (index, score) in enumerate(zip(top_k_indices, top_k_scores)):
        print(f"{i}. {df_paris.iloc[index]['hotel_name']} ")
        print(f"Review:{df_paris.iloc[index]['review_text']} ")
        print(f"Distance: {score:.4f}")
        print()
    

if __name__ == "__main__":
    main()