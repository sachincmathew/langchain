import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")

# Ensure the text file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
    )

# Check file permissions
if not os.access(file_path, os.R_OK):
    raise PermissionError(f"The file {file_path} is not readable. Please check the file permissions.")

# Read the text content from the file
loader = TextLoader(file_path, encoding='utf-8')
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Display information about the split documents
print("\n--- Document Chunks Information ---")
print(f"Number of document chunks: {len(docs)}")
print(f"==================================")
print(f"Sample chunk:\n{docs[0].page_content}\n")
print(f"==================================")

# Create embeddings
print("\n--- Creating embeddings ---")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)  # Update to a valid embedding model if needed
print("\n--- Finished creating embeddings ---")

# Create the vector store and persist it automatically
print("\n--- Creating vector store ---")
index_name = "default"
db = AstraDBVectorStore.from_documents(
    docs, embeddings, collection_name="db_default")
print("\n--- Finished creating vector store ---")