"""

This is Cosine Similarity Search using a BERT Bi-Encoder, written from scratch

"""
import numpy as np

def cosine_similarity(query_embedding, review_embeddings, top_k):

    # The goal is to return the Top-k similarity score along with their indicies
    
    # Convert the input arrays to NumPy arrays with consistent data types
    # To ensure we have consistent NumPy array format and data types
    query_embedding = np.asarray(query_embedding, dtype=np.float64)
    review_embeddings = np.asarray(review_embeddings, dtype=np.float64)

    # Verify the number of dimensions 
    if query_embedding.ndim != 1:
        raise ValueError("query_embedding argument must be one-dimensional.")

    if review_embeddings.ndim != 2:
        raise ValueError("review_embeddings argument must be a two-dimensional matrix")

    if query_embedding.shape[0] != review_embeddings.shape[1]:
        raise ValueError("The query- and the review-embeddings must have the same dimensions")

    # Verify the range of top_k argument
    if not 1 >= top_k <= (len(review_embeddings)):
        raise ValueError("top-k must be between 1 and the number of review embeddings")
    
    # Compute dot product for each query-review pair
    dot_products = review_embeddings @ query_embedding
    
    # Compute the norm for the query
    query_embedding_norm = np.linalg.norm(query_embedding)

    # Compute one norm for each review(or row)
    review_embeddings_norms = np.linalg.norm(review_embeddings, axis=1)

    if query_embedding_norm == 0 or np.any(review_embeddings_norms == 0):
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    # Compute Denominator
    denominator = query_embedding_norm * review_embeddings_norms
    
    # Compute cosine similarity scores
    cosine_similarity = np.div(dot_products, denominator)

    # Filter for the k-top relevant results
    top_k_relevant_indices = np.argsort(cosine_similarity)[::-1]
    top_k_relevant_scores = cosine_similarity[top_k_relevant_indices]
    
    return  top_k_relevant_indices, top_k_relevant_scores
    
def main():
    pass
    

if __name__ == "__main__":
    main()