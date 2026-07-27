from sentence_transformers import SentenceTransformer
import scipy.spatial
import pandas as pd

#takes user + documents embeddings + the actual documents
def semantic_search(query_embedding, embeddings, examples):
    
    #scores computes the cosine distance, which is different than the cosine similiarty
        # cosine similiarty = cos(θ) 
        # cosine distance =  1 - cos(θ)
    scores = [scipy.spatial.distance.cosine(query_embedding, doc) for doc in embeddings]

    for i, doc in enumerate(examples):
        print(f"Example {i}: {examples[i]}")
        print(f"Score: {1 - scores[i]:.4f}")
        
    print("Most similar example:", examples[scores.index(min(scores))])

def main():
    examples = [ #A
    "The cat is playing in the garden",
    "A dog and a cat are good pets",
    "Cats love to chase mice",
    "Machine learning is based on algorithms",
    "Deep learning uses neural networks",
    "Recurrent networks have connections"
    ]

    #load encoder 
        #is 'all-MiniLM-L6-v2' the encoder's name by default ?  No, we need to specify which encoder to use everytime
    model = SentenceTransformer('all-MiniLM-L6-v2')

    #create embeddings for the examples
    embeddings = model.encode(examples)

    #embed user query
    query = "Machine Learning"
    query_embedding = model.encode(query) #shape: (384,)
    
    semantic_search(query_embedding, embeddings, examples)

if __name__ == "__main__":
    main()