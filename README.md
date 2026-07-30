### Hotel Search Engine

🚧 **Work in Progress**

A small semantic-search project that finds hotel reviews most similar to a natural-language query.

The project uses:

- Sentence Transformers to create text embeddings
- NumPy to calculate cosine similarity
- Hugging Face Datasets to load the hotel-review dataset
- pandas to filter and clean the data

Example query

query = "Hotel near the Louvre with great food nearby."

Here’s a screenshot of the responses:
 ![result on terminal](https://i.ibb.co/BHSfshnk/Screenshot-2026-07-30-at-03-18-22.png)

Todos:
- Creating a user interface
- Supporting different cities and search queries
- Saving embeddings to avoid recomputing them
- Reuse cached embeddings between searches
- Choose a Licence for the project
- Host the project
