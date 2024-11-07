# doc_processor.py
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from typing import Dict, List, Optional
import logging
from pathlib import Path

class CarbonDocProcessor:
    def __init__(self, json_path: str, vector_db_path: str):
        """
        Initialize the Carbon documentation processor.
        
        Args:
            json_path: Path to the ibm_carbon_v1.json file
            vector_db_path: Path where the vector database will be stored
        """
        self.json_path = Path(json_path)
        self.vector_db_path = Path(vector_db_path)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Validate input file
        self._validate_input_file()
    
    def _validate_input_file(self) -> None:
        """Validate that the input file exists and has the correct format."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"Could not find file: {self.json_path}")
            
        if self.json_path.name != "ibm_carbon_v1.json":
            raise ValueError(
                f"Expected file name 'ibm_carbon_v1.json', got '{self.json_path.name}'"
            )
            
        # Validate file size
        if self.json_path.stat().st_size == 0:
            raise ValueError("JSON file is empty")
    
    def _validate_carbon_format(self, data: Dict) -> bool:
        """
        Validate that the JSON data follows the expected Carbon documentation format.
        Returns True if valid, raises ValueError if invalid.
        """
        required_fields = ['url', 'title', 'content']
        
        if not isinstance(data, list):
            raise ValueError("JSON data must be a list of documentation entries")
            
        for entry in data:
            if not isinstance(entry, dict):
                raise ValueError("Each entry must be a dictionary")
                
            missing_fields = [field for field in required_fields if field not in entry]
            if missing_fields:
                raise ValueError(
                    f"Entry missing required fields: {', '.join(missing_fields)}"
                )
        
        return True
    
    def load_documents(self) -> List[Dict]:
        """Load and validate documents from the Carbon JSON file."""
        self.logger.info(f"Loading documents from {self.json_path}")
        
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate format
            self._validate_carbon_format(data)
            
            self.logger.info(f"Successfully loaded {len(data)} documents")
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            self.logger.error(f"Error loading documents: {e}")
            raise
    
    def process_documents(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> Optional[Chroma]:
        """
        Process Carbon documentation and create vector store.
        
        Args:
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            
        Returns:
            Chroma vector store instance or None if processing fails
        """
        try:
            # Load documents
            documents = self.load_documents()
            
            # Create document chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                add_start_index=True
            )
            
            # Prepare documents for splitting
            texts = []
            metadatas = []
            
            for doc in documents:
                chunks = text_splitter.create_documents(
                    texts=[doc['content']],
                    metadatas=[{
                        'url': doc['url'],
                        'title': doc['title'],
                        'source': 'IBM Carbon Design System'
                    }]
                )
                
                for chunk in chunks:
                    texts.append(chunk.page_content)
                    metadatas.append(chunk.metadata)
            
            self.logger.info(f"Created {len(texts)} chunks from {len(documents)} documents")
            
            # Create vector store
            vectorstore = Chroma.from_texts(
                texts=texts,
                metadatas=metadatas,
                embedding=OpenAIEmbeddings(),
                persist_directory=str(self.vector_db_path)
            )
            
            # Persist the vector store
            vectorstore.persist()
            self.logger.info(f"Vector store created and persisted at {self.vector_db_path}")
            
            return vectorstore
            
        except Exception as e:
            self.logger.error(f"Error processing documents: {e}")
            raise
    
    def get_document_stats(self) -> Dict:
        """Get statistics about the processed documents."""
        try:
            documents = self.load_documents()
            
            total_content_length = sum(len(doc['content']) for doc in documents)
            
            return {
                'total_documents': len(documents),
                'total_content_length': total_content_length,
                'average_content_length': total_content_length / len(documents),
                'unique_urls': len(set(doc['url'] for doc in documents)),
                'file_size_mb': round(self.json_path.stat().st_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting document stats: {e}")
            raise

processor = CarbonDocProcessor(
    json_path="ibm_carbon_v1.json",
    vector_db_path="./carbon_vector_db"
)

# Get stats about the documentation
stats = processor.get_document_stats()
print("Documentation statistics:")
for key, value in stats.items():
    print(f"{key}: {value}")

# Process the documents
vectorstore = processor.process_documents()