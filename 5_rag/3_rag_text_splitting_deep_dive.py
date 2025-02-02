import os

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Define the directory containing the text file
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "romeo_and_juliet.txt")

# Read the text content from the file
loader = TextLoader(file_path, encoding='utf-8')
documents = loader.load()
documents[0].metadata = {"source": "Romeo and Juliet by William Shakespeare"}  # Add metadata

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # Update to a valid embedding model if needed


# # Function to create and persist vector store
# def create_vector_store(docs, index_name):
#     db = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
#     print(f"--- Finished creating vector store in index {index_name} ---")

# # 1. Character-based Splitting
# # Splits text into chunks based on a specified number of characters.
# # Useful for consistent chunk sizes regardless of content structure.
# print("\n--- Using Character-based Splitting ---")
# char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# char_docs = char_splitter.split_documents(documents)
# create_vector_store(char_docs, "demo-character-text-splitter")

# # 2. Sentence-based Splitting
# # Splits text into chunks based on sentences, ensuring chunks end at sentence boundaries.
# # Ideal for maintaining semantic coherence within chunks.
# print("\n--- Using Sentence-based Splitting ---")
# sent_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
# sent_docs = sent_splitter.split_documents(documents)
# create_vector_store(sent_docs, "demo-sentence-transformers-t-t-s")

# # 3. Token-based Splitting
# # Splits text into chunks based on tokens (words or subwords), using tokenizers like GPT-2.
# # Useful for transformer models with strict token limits.
# print("\n--- Using Token-based Splitting ---")
# token_splitter = TokenTextSplitter(chunk_overlap=0, chunk_size=512)
# token_docs = token_splitter.split_documents(documents)
# create_vector_store(token_docs, "demo-token-text-splitter")

# # 4. Recursive Character-based Splitting
# # Attempts to split text at natural boundaries (sentences, paragraphs) within character limit.
# # Balances between maintaining coherence and adhering to character limits.
# print("\n--- Using Recursive Character-based Splitting ---")
# rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# rec_char_docs = rec_char_splitter.split_documents(documents)
# create_vector_store(rec_char_docs, "demo-recursive-character-text-splitter")

# # 5. Custom Splitting
# # Allows creating custom splitting logic based on specific requirements.
# # Useful for documents with unique structure that standard splitters can't handle.
# print("\n--- Using Custom Splitting ---")
# class CustomTextSplitter(TextSplitter):
#     def split_text(self, text):
#         # Custom logic for splitting text
#         return text.split("\n\n")  # Example: split by paragraphs

# # ##### Not executing as Pinecine free tier only allows 5 indexes #####
# # custom_splitter = CustomTextSplitter()
# # custom_docs = custom_splitter.split_documents(documents)
# # create_vector_store(custom_docs, "demo_CustomTextSplitter")


# Function to query a vector store
def query_vector_store(index_name, query):
    db = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )
    relevant_docs = retriever.invoke(query)
    print(f"\n------------------------------{index_name}------------------------------")
    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")


# Define the user's question
query = "How did Juliet die?"

# Query each vector store
query_vector_store("demo-character-text-splitter", query)
query_vector_store("demo-sentence-transformers-t-t-s", query)
query_vector_store("demo-token-text-splitter", query)
query_vector_store("demo-recursive-character-text-splitter", query)
#query_vector_store("demo_CustomTextSplitter", query)
