import os

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
index_name = "books-rag-demo"
db = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Define the user's question
query = "Who is Odysseus wife?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.5},
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
