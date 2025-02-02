import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Define the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")

print(f"Books directory: {books_dir}")

# List all text files in the directory
book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

# Read the text content from each file and store it with metadata
documents = []
for book_file in book_files:
    file_path = os.path.join(books_dir, book_file)
    loader = TextLoader(file_path, encoding='utf-8')

    #returns a list of Document where the content of the document is inside page_content attribute
    book_docs = loader.load()
    for doc in book_docs:
        # Add metadata to each document indicating its source
        doc.metadata = {"source": book_file}
        documents.append(doc)
        # print(doc.metadata)
print(f"Number of documents: {len(documents)}")

# # Split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# # Display information about the split documents
print("\n--- Document Chunks Information ---")
print(f"Number of document chunks: {len(docs)}")

# Create the vector store and persist it automatically
print("\n--- Creating vector store ---")
index_name = "books-rag-demo"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
print("\n--- Finished creating vector store ---")