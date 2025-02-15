import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Step 1: Scrape the content from apple.com using WebBaseLoader
# # WebBaseLoader loads web pages and extracts their content
# urls = ["https://www.apple.com/"]

# # Create a loader for web content
# loader = WebBaseLoader(urls)
# documents = loader.load()

# # Step 2: Split the scraped content into chunks
# # CharacterTextSplitter splits the text into smaller chunks
# def split_text(documents: list, chunk_size: int = 1000, chunk_overlap: int = 0) -> list:
#     text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     return text_splitter.split_documents(documents)

# docs = split_text(documents)


# # Display information about the split documents
# print("\n--- Document Chunks Information ---")
# print(f"Number of document chunks: {len(docs)}")
# # print(f"Sample chunk:\n{docs[0].page_content}\n")

# Step 3: Create embeddings for the document chunks
# OpenAIEmbeddings turns text into numerical vectors that capture semantic meaning
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Step 4: Create and persist the vector store with the embeddings
index_name = "webscrape"

##### Split into two phases as we dont need to always create the vector store (only once)
# Phase 1: Create a Pinecone vector store with the document chunks and embeddings
# db = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)

# Phase 2: Load the existing vector store with the embedding function
db = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Step 5: Query the vector store
# Create a retriever for querying the vector store
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Define the user's question
query = "What new products are announced on Apple.com?"

# Retrieve relevant documents based on the query
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
