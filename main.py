# Install & Import required libraries
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import torch

# Goal: Output the top-k most relevant results

def main():
    
    # Define user query
    user_query = "An affordable Hotel with a view of a the Eiferl tower"

    k = 5

    # Load dataset
    dataset = load_dataset("traversaal-ai-hackathon/hotel_datasets")
    
    # select the 'train' split & turn it into a DataFrame
    df = dataset["train"].to_pandas()
    
    # Filter for hotels only in Parais & clean dataset
    df_paris = df.loc[
        (df.locality == "Paris")
        & (df.review_text.notna() )
        & (df.review_text.str.strip() != "")
    ]
    
    # Extract reviews
    reviews = df_paris.review_text.tolist()
    
    # Load an Embedding model
    model = SentenceTransformer("all-MiniLM-L6-V2")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Model is moved to {device}")
    
    # Encode user query  & make sure Embeddings are of type float32
    query_embedding = model.encode([user_query]).astype(np.float32)
    
    # Encode candidate documents 
    review_embeddings = model.encode(reviews, show_progress_bar=True).astype(np.float32)
    
    # Perform Similarity Search using FAISS
    # Check dimensions 
    assert review_embeddings[1] == query_embedding[1]
    
    # Get embedding dimension
    embedding_dimension = review_embeddings.shape[1]
    
    # Create a FAISS index
    index = faiss.IndexFlatIP(embedding_dimension)

    # Add review embeddings
    index.add(review_embeddings)
    similarity_scores, indices = index.search(query_embedding, k)
    
    
    # Output top-k most relevant results
    for i, (score, idx) in enumerate(zip(similarity_scores, indices)):
        row = df_paris.iloc[idx]
        
        print(f"Query: {user_query}")
        print(f"{i}. {row['hotel_name']}")
        print(f"Review: {row['review_text']} ")
        print(f"Distance: {score}")
        print()
              

if __name__ == "__main__":
    main()