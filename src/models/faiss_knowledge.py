

# faiss_knowledge.py
import os
import pickle
import logging
import faiss
import numpy as np
from typing import List, Tuple, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import fitz  # PyMuPDF

# Configure logging
logger = logging.getLogger(__name__)

class FAISSKnowledgeBase:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize FAISS knowledge base with HuggingFace embeddings.
        
        Args:
            embedding_model (str): Name of the embedding model to use
        """
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            self.index = None
            self.texts = []
            self.index_path = "faiss_index"
            logger.info(f"FAISSKnowledgeBase initialized with model: {embedding_model}")
        except Exception as e:
            logger.exception(f"Initialization failed: {str(e)}")
            raise

    def load_pdfs(self, pdf_path: str) -> List[str]:
        """
        Load and process a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[str]: List of text chunks from the PDF
        """
        try:
            # Open the PDF file using PyMuPDF
            doc = fitz.open(pdf_path)
            
            # Extract text from each page
            texts = []
            for page in doc:
                text = page.get_text()
                if text:
                    texts.append(text)
            
            # Combine all text into a single string
            combined_text = " ".join(texts)
            
            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            text_chunks = text_splitter.split_text(combined_text)
            
            logger.info(f"Successfully processed PDF: {len(text_chunks)} chunks created")
            return text_chunks
            
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            return []

    def build_index(self, texts: List[str]) -> None:
        """
        Create a FAISS index from text chunks.
        
        Args:
            texts (List[str]): List of text chunks to index
        """
        try:
            self.index = FAISS.from_texts(texts, self.embeddings).index
            self.texts = texts
            logger.info("FAISS index built successfully")
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}")
            raise

    def save_index(self) -> None:
        """Save FAISS index and associated texts to disk."""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(f"{self.index_path}_texts.pkl", "wb") as f:
                pickle.dump(self.texts, f)
            logger.info("FAISS index and texts saved successfully")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise

    def load_index(self) -> bool:
        """
        Load FAISS index and associated texts from disk.
        
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                with open(f"{self.index_path}_texts.pkl", "rb") as f:
                    self.texts = pickle.load(f)
                logger.info("FAISS index and texts loaded successfully")
                return True
            logger.warning("No FAISS index found")
            return False
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """
        Search FAISS index and return top_k results.
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            
        Returns:
            List[str]: List of relevant text chunks
        """
        try:
            if self.index is None:
                logger.error("FAISS index is not initialized")
                return []
            
            # Convert query to vector
            query_vector = np.array(self.embeddings.embed_query(query)).reshape(1, -1)
            
            # Perform the search
            distances, indices = self.index.search(query_vector, top_k)
            
            # Return the results
            return [self.texts[i] for i in indices[0] if i != -1]
            
        except Exception as e:
            logger.error(f"Error searching FAISS index: {str(e)}")
            return []

def initialize_knowledge_base(pdf_path: str) -> Optional[FAISSKnowledgeBase]:
    """
    Initialize and set up the knowledge base.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        Optional[FAISSKnowledgeBase]: Initialized knowledge base or None if initialization fails
    """
    try:
        kb = FAISSKnowledgeBase()
        
        # Try to load existing index
        if not kb.load_index():
            # If no index exists, create new one
            text_chunks = kb.load_pdfs(pdf_path)
            kb.build_index(text_chunks)
            kb.save_index()
        
        return kb
    except Exception as e:
        logger.exception("Failed to initialize knowledge base")
        return None