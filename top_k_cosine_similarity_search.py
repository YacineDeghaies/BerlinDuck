import numpy as np
import pandas as pd
import scipy.spatial as sp

def cosine_similarity(query_embedding, review_embeddings, k):
    # We first flatten the query embedding to be able to use it for the dot product
    query = np.asarray(query_embedding, dtype=float).reshape(-1)
    
    # Make sure review_embeddings is an ndarray with a float type
    review = np.asarray(review_embeddings, dtype=float)
    
    # Make sure review is a 2-D ndarray
    if review.ndim != 2:
        raise ValueError(
            "review_embeddings: must have shape "
            "(n_reviews, embedding_dim)"
        )
    
    # Verify if dimensions of embedding match
    if review.shape[1] != query.size:
        raise ValueError(
            f"Embedding dimensions do not match: query has dimensions"
            f"{query.size}, while reviews has dimensions"
            f"{review.shape[1]}"
        )

    # Verify the k parameter
    if (
        isinstance(k, (bool, np.bool_)) or
        not isinstance(k, (int, np.integer)) or
        k < 1
    ):
        raise ValueError("k must be a positive integer."

    # Calculate the dot products between the query embedding and each review embedding
    dot_products = np.dot(review_embeddings, query_embedding)

    # Normalize the query embedding
    query_embedding_norm = np.linalg.norm(query_embedding)

    # Normalize the review embeddings
        # axis=1 means across the columns not the rows
    review_embeddings_norm = np.lingalg.norm(query_embedding, axis=1)
    
    # Calculate the cosine similarity
        # before that we need to compute the Norms of the embeddings
    
    cosine_similarity = dot_products / (query_embedding_norm * review_embeddings_norm )
    
    # To retrieve the top-k similarity scores, we need their indicies first
    # We want to get the top-k largest cosine similarity scores, not the top-k smallest ones
    # Since, .argsort() sorts in asc order, we need to reverse the order of numbers by negating the similarity scores
    top_k_indicies = np.argsort(-cosine_similarity)[:k]
    
    # Retrieve the top-k cosine similarity scores using their indicies
    top_k_similarity_scores = [cosine_similarity[i] for i in top_k_indicies]
    
    #return top_k_indicies, top_k_cosine_similarity_scores
    return top_k_indicies, top_k_similarity_scores
    