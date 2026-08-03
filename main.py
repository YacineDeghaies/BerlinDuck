import numpy as np

# a = np.array([1,2,3], dtype=np.float64)
# a = np.array([1.1,3.14, 9.1415], dtype=np.float64)

# try:
#     code goe shere
# except ValueFormat:
#     code goes here
    
# except DivisionZeroError:
#     code goes here
# except TypeError:
#     code goes here
# except SyntaxError:
#     code goes here
# except RunTimeError:
#     code goes here
# except OverflowError:
#     code goes here
# except:
#     code goes here


query_embedding = np.array([1.0, 2.0, 3.0])  # Shape: (3,)

review_embeddings = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
])  # Shape: (2, 4)

print(query_embedding.shape[0])      # 3
print(review_embeddings.shape[1])    # 4

if query_embedding.ndim != 1:
    raise ValueError("query_embedding must be one-dimensional.")

if review_embeddings.ndim != 2:
    raise ValueError("review_embeddings must be a two-dimensional matrix.")

print(query_embedding.shape[0] != review_embeddings.shape[1])
# True
print(query_embedding)


query_embedding = np.squeeze(query_embedding)
print(query_embedding)
