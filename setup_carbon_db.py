from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import json
import os

# Load environment variables
load_dotenv()

def setup_vector_db(json_path="ibm_carbon_v1.json", db_path="./vector_db"):
    """
    Create a vector database from IBM Carbon JSON documentation.
    This is the 'R' (Retrieval) part of RAG.
    """
    print(f"Loading Carbon documentation from {json_path}...")
    
    # Load the JSON file
    with open(json_path, 'r') as f:
        docs = json.load(f)
        
    # Prepare the documents
    texts = []
    metadatas = []
    
    for url, content in docs.items():
        texts.append(content)
        metadatas.append({
            'title': url.split('/')[-1],  # Use the last part of the URL as title
            'url': url
        })
    
    print(f"Processing {len(texts)} documents...")
    
    # Create and persist the vector store
    vectorstore = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=OpenAIEmbeddings(),
        persist_directory=db_path
    )
    
    print(f"Vector database created at {db_path}")
    return vectorstore

if __name__ == "__main__":
    setup_vector_db()

