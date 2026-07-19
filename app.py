from sentence_transformers import SentenceTransformer
import scipy.spatial
import pandas as pd

examples = [ #A
"The cat is playing in the garden",
"A dog and a cat are good pets",
"Cats love to chase mice",
"Machine learning is based on algorithms",
"Deep learning uses neural networks",
"Recurrent networks have connections"]

#Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')
print(model)


