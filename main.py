import numpy as np
# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("all-MiniLM-L6-v2")

# empty_embedding = model.encode([""])
# print(np.linalg.norm(empty_embedding[0]))

indices = [10, 20, 30]
scores = [0.9, 0.8, 0.7]
pairs = zip(indices, scores)
print(pairs.__next__())