import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

def validate_file(file_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )
    if not os.access(file_path, os.R_OK):
        raise PermissionError(
            f"The file {file_path} is not readable. Please check the file permissions."
        )

def load_document(file_path: str) -> list:
    loader = TextLoader(file_path, encoding='utf-8')
    return loader.load()

def split_text(documents: list, chunk_size: int = 1000, chunk_overlap: int = 0) -> list:
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def main():
    # Define file path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "books", "odyssey.txt")
    
    # Process document
    validate_file(file_path)
    documents = load_document(file_path)
    text_chunks = split_text(documents)
    
    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(text_chunks)}")
    print(f"==================================")
    print(f"Sample chunk:\n{text_chunks[0].page_content}\n")
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
    db = PineconeVectorStore.from_documents(
        text_chunks, embeddings, index_name=index_name)
    print("\n--- Finished creating vector store ---")

if __name__ == "__main__":
    main()