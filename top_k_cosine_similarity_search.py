import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

def cosine_similarity(query_embedding, review_embeddings, k):
    # We first flatten the query embedding and make sure it's of type float
    query = np.asarray(query_embedding, dtype=float).reshape(-1)
    
    # Make sure review_embeddings is an floating-point ndarray
    reviews = np.asarray(review_embeddings, dtype=float)
    
    # Make sure review is a 2-D ndarray of shape:
    # (number_of_reviews, embedding_dimension)
    if reviews.ndim != 2:
        raise ValueError(
            "review_embeddings: must have shape "
            "(n_reviews, embedding_dim)"
        )
    
    # The query and reviews must have the same embedding dimension.
    if reviews.shape[1] != query.size:
        raise ValueError(
            f"Embedding dimensions do not match: query has dimensions"
            f"{query.size}, while reviews has dimensions"
            f"{reviews.shape[1]}"
        )

    # k must be a positive integer, 
    if (
        isinstance(k, (bool, np.bool_)) or
        not isinstance(k, (int, np.integer)) or
        k < 1
    ):
        raise ValueError("k must be a positive integer.")

    # Verify against empty collections
    if reviews.shape[0] == 0:
        raise ValueError("review_embeddings must contain atleast one reviews")

    # Do not request more results than there are reviews.
    k = min(k, reviews.shape[0])

    # Calculate norms
    query_norm = np.linalg.norm(query)
    reviews_norm = np.linalg.norm(reviews, axis=1)
    
    denominators = query_norm  * reviews_norm 
    
    # One dot product for each review
    dot_products = reviews @ query

    # Calculate the cosine similarity
    similarities = np.divide(
        dot_products,
        denominators,
        out=np.zeros_like(dot_products, dtype=float), # decides what value the skipped positions should contain
        where=denominators != 0, # this is mask that decides which position should we calculate and which one should we not
    )
    
    # To retrieve the top-k similarity scores, we need their indicies first
    # We want to get the top-k largest cosine similarity scores, not the top-k smallest ones
    # Since, .argsort() sorts in asc order, we need to reverse the order of numbers by negating the similarity scores
    top_k_indicies = np.argsort(-similarities)[:k]
    
    # Retrieve the top-k similarity scores using their indicies
    top_k_similarity_scores = similarities[top_k_indicies]
    
    return top_k_indicies, top_k_similarity_scores

def main():
    
    # Load an embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Define an example search query
    query =  "Hotel near the Louvre with great food nearby."

    # Load the datasetset
    dataset = load_dataset("traversaal-ai-hackathon/hotel_datasets")

    # Convert it into a DataFrame
    df = dataset["train"].to_pandas()

    # Filter for hotels in Paris only
    df_paris = df.loc[df.locality == "Paris"].copy()
    
    # Show how many unique hotels in paris the dataset has
    print(df_paris.hotel_name.value_counts())
    
    # Clean the data
    df_paris = df_paris.dropna(subset=["review_text"])
    # this means: if after stripping the whitespaces we still have a value e.g., "a" or something
    df_paris = df_paris.loc[df_paris.review_text.str.strip() != ""]

    #resetting the index
    df_paris = df_paris.reset_index(drop=True)

    # Extract reviews
    reviews = df_paris.review_text.tolist()
    
    # Create embeddings for the reviews
    reviews_embeddings = model.encode(reviews, show_progress_bar=True)
    
    print(f"Embeddings shape: {reviews_embeddings.shape}")

    # Embed the search query
    query_embedding = model.encode([query])

    # Top-k similar reviews to retrieve
    k = 5

    indices, scores = cosine_similarity(query_embedding, reviews_embeddings, k)
    
    print(f"Query: {query}")
    print("Top hotel with similar reviews matching the query")
    for i, (idx, score) in enumerate(zip(indices, scores), 1):
        print(f"{i}. {df_paris.iloc[idx]['hotel_name']}")  # D
        print(f"Review: {df_paris.iloc[idx]['review_text']}")  # E
        print(f"Cosine similarity: {score:.4f}")  # F
        print()
        
if __name__ == "__main__":
    main()