import os

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Define the index name
index_name = "books-rag-demo"

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Function to query a vector store with different search types and parameters
def query_vector_store(index_name, query, embedding_function, search_type, search_kwargs):
    db = PineconeVectorStore(index_name=index_name, embedding=embedding_function)
    retriever = db.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    relevant_docs = retriever.invoke(query)
    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")


# Define the user's question
query = "How did Juliet die?"

# Showcase different retrieval methods

# 1. Similarity Search
# This method retrieves documents based on vector similarity.
# It finds the most similar documents to the query vector based on cosine similarity.
# Use this when you want to retrieve the top k most similar documents.
print("\n--- Using Similarity Search ---")
query_vector_store(index_name, query, embeddings, "similarity", {"k": 3})

# 2. Max Marginal Relevance (MMR)
# This method balances between selecting documents that are relevant to the query and diverse among themselves.
# 'fetch_k' specifies the number of documents to initially fetch based on similarity.
# 'lambda_mult' controls the diversity of the results: 1 for minimum diversity, 0 for maximum.
# Use this when you want to avoid redundancy and retrieve diverse yet relevant documents.
# Note: Relevance measures how closely documents match the query.
# Note: Diversity ensures that the retrieved documents are not too similar to each other,
#       providing a broader range of information.
print("\n--- Using Max Marginal Relevance (MMR) ---")
query_vector_store(index_name, query, embeddings, "mmr", {"k": 3, "fetch_k": 20, "lambda_mult": 0.5})

# 3. Similarity Score Threshold
# This method retrieves documents that exceed a certain similarity score threshold.
# 'score_threshold' sets the minimum similarity score a document must have to be considered relevant.
# Use this when you want to ensure that only highly relevant documents are retrieved, filtering out less relevant ones.
print("\n--- Using Similarity Score Threshold ---")
query_vector_store(index_name, query, embeddings, "similarity_score_threshold", {"k": 3, "score_threshold": 0.1})

print("Querying demonstrations with different search types completed.")
